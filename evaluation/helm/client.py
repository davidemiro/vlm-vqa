from threading import Lock
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoProcessor

from helm.common.cache import CacheConfig
from helm.common.images_utils import open_image
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.media_object import TEXT_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, GeneratedOutput, Token, wrap_request_time
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt
from helm.tokenizers.tokenizer import Tokenizer


from vlm.configuration_vlm import VLMConfig
from vlm.modeling_vlm import VLMForCausalLM
from vlm.processing_vlm import VLMProcessor
from transformers import AutoTokenizer
from vlm.load_fsdp_model import load_model


import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)




try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["images"])

# Added to solve: cutlassF: no kernel found to launch!
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


class LoadedVLMForCausalLM:
    """Loaded model and processor for PaliGemma."""

    model: VLMForCausalLM
    processor: AutoProcessor

    def __init__(self, model: VLMForCausalLM, processor: AutoProcessor):
        self.model = model
        self.processor = processor


_models_lock: Lock = Lock()
_models: Dict[str, Optional[VLMForCausalLM]] = {}


class VLMClient(CachingClient):
    """
    VLM is a lightweight vision-language model inspired by PaliGemma and based on open components such
    as DinoV2 and Gemma2 language model.
    It takes both image and text as input and generates text as output, supporting english languages.
    It has been trained specifically for Visual Question Answering task.
    """

    def __init__(self, tokenizer: Tokenizer, tokenizer_name: str, cache_config: CacheConfig, token=""):
        super().__init__(cache_config=cache_config)
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        self.tokenizer_name = tokenizer_name
        self._device = "mps"
        self.token = token

    def _get_model(self, checkpoint: str) -> LoadedVLMForCausalLM:
        global _models_lock
        global _models

        # Ensure that only one thread is loading the model at a time
        with _models_lock:
            if checkpoint not in _models or _models[checkpoint] is None:
                hlog(f"Loading model {checkpoint} and caching in memory...")

                fsdp_path = ""

                vlm_config = VLMConfig(text_length=64, num_patches=257, visual_embed_dim=768)
                processor = VLMProcessor(vlm_config, token=self.token)
                model = load_model(fsdp_path).eval()
                model = model.to(self._device)

                _models[checkpoint] = LoadedVLMForCausalLM(model, processor)
            loaded_model_processor = _models[checkpoint]

        assert loaded_model_processor is not None
        return loaded_model_processor

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        loaded_model_processor = self._get_model(request.model_deployment)
        model = loaded_model_processor.model
        processor = loaded_model_processor.processor


        images = None
        prompt_pieces: List[str] = []
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                images = open_image(media_object.location).convert("RGB")
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                prompt_pieces.append(media_object.text)
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")
        prompt_text: str = "".join(prompt_pieces)

        model_inputs = processor(text=prompt_text, image=images, return_tensors="pt")

        input_len = model_inputs["input_ids"].shape[-1]

        model_inputs = {k: v.to(self._device) for k, v in model_inputs.items()}

        completions: List[GeneratedOutput] = []
        with htrack_block(f""):
            try:
                concat_results = []
                for i_completion in range(request.num_completions):

                    def do_it() -> Dict[str, Any]:

                        with torch.no_grad():

                            generation = model.generate(
                                **model_inputs, do_sample=False, use_cache=False
                            )[0]
                            if not request.echo_prompt:
                                generation = generation[input_len:]


                            decoded = self.tokenizer.decode(generation)
                            decoded = decoded[:decoded.index("<eos>")]
                            decoded = decoded.replace("<bos>","").replace("<eos>","")

                            del generation
                            return {"output": decoded, "prompt": prompt_text}

                    # Include the prompt and model name in the cache key
                    cache_key = CachingClient.make_cache_key(
                        raw_request={
                            "n": request.num_completions,
                            "i": i_completion,
                            "model": request.model,
                            "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                        },
                        request=request,
                    )
                    result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
                    concat_results.append(result)

            except RuntimeError as model_error:
                return RequestResult(success=False, cached=False, error=str(model_error), completions=[], embedding=[])

            for result in concat_results:
                text = result["output"]
                hlog(f"Prompt: {result['prompt']}")
                hlog(f"Generated text: {text}")
                raw_tokens = self.tokenizer.tokenize(text)
                tokens = [ Token(text=raw_token, logprob=0) for raw_token in raw_tokens ]
                completions.append(GeneratedOutput(text=text, logprob=0, tokens=tokens))


        del model_inputs
        return RequestResult(
            success=True,
            cached=cached,
            request_time=result["request_time"],
            completions=completions,
            embedding=[],
        )
