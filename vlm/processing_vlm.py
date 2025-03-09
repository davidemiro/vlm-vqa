from transformers import AutoProcessor, AutoImageProcessor, AutoTokenizer, ProcessorMixin
from vlm.configuration_vlm import VLMConfig
import torch


class VLMProcessor(ProcessorMixin):
    def __init__(self, config: VLMConfig, token: str, **kwargs) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it', token=token)
        self.feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=True, token=token)
        self.context_length = config.context_length
        self.chat_template = None

    def _training_processing(self, text, image, label, return_tensors="pt"):
        text = "<bos>" + text + label
        text_tokenized = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.context_length,
                                        return_tensors=return_tensors)

        label_tokenized = self.tokenizer(label, truncation=True, padding="max_length", max_length=self.context_length,
                                         return_tensors=return_tensors)['input_ids']

        label_tokenized[label_tokenized == 0] = -100

        pixel_values = torch.tensor(self.feature_extractor(images=image, return_tensors="np")['pixel_values'],
                                    requires_grad=True, dtype=torch.float16)

        return {'input_ids': text_tokenized['input_ids'], 'attention_mask': text_tokenized['attention_mask'],
                'labels': label_tokenized, 'pixel_values': pixel_values}

    def _inference_processing(self, text, image, return_tensors="np"):
        text = "<bos>" + text
        text_tokenized = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.context_length,
                                        return_tensors=return_tensors)

        pixel_values = torch.tensor(self.feature_extractor(images=image, return_tensors="np")['pixel_values'],
                                    requires_grad=True, dtype=torch.float16)

        return {'input_ids': text_tokenized['input_ids'], 'attention_mask': text_tokenized['attention_mask'],
                'pixel_values': pixel_values}

        def batch_decode(self, *args, **kwargs):
            return self.tokenizer.batch_decode(*args, **kwargs)

        def decode(self, *args, **kwargs):
            return self.tokenizer.decode(*args, **kwargs)

        @property
        def model_input_names(self):
            tokenizer_input_names = self.tokenizer.model_input_names
            image_processor_input_names = self.feature_extractor.model_input_names
            return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def __call__(self, text, image, label=None, return_tensors="pt"):
        if label is None:
            return self._inference_processing(text, image, return_tensors=return_tensors)
        else:
            return self._training_processing(text, image, label, return_tensors=return_tensors)



