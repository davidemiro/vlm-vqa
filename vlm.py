from typing import Optional, Union, Tuple

import transformers
from torch import nn
import torch
from transformers import Gemma2ForCausalLM, HybridCache, DataCollatorForLanguageModeling
from transformers.modeling_outputs import CausalLMOutputWithPast

IMAGE_TOKEN = "<image>"


def vlm_tokenizer(tokenizer: transformers.GemmaTokenizer):

    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    return tokenizer


class VLMDataCollator(DataCollatorForLanguageModeling):

    def __init__(self, tokenizer: transformers.GemmaTokenizer):
        self.tokenizer = tokenizer



class VLMGemma2ForCausalLM(Gemma2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.linear_projector = nn.Linear(config.visual_embed_dim, config.hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        text_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        visual_embeds = self.linear_projector(visual_embeds)
        input_embeds = torch.cat((visual_embeds, text_embeds), dim=1)

        return super().forward(input_ids,attention_mask, position_ids, past_key_values, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, num_logits_to_keep)










