from typing import Optional, Union, Tuple
from torch import nn
import torch
from transformers import Gemma2ForCausalLM, HybridCache, Dinov2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from configuration_vlm import VLMConfig
from processing_vlm import VLMProcessor


class VLMForCausalLM(Gemma2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.linear_projector = nn.Linear(config.visual_embed_dim, config.hidden_size)
        self.vit = Dinov2Model(config=config.vit_config)

        self.image_token_id = self.config.image_token_id
        self.pad_token_id = self.config.pad_token_id

        self.embed_tokens = nn.Embedding(self.config.vocab_size + 1, self.config.hidden_size)

        with torch.no_grad():
            self.embed_tokens.weight[:self.config.vocab_size, :] = self.model.embed_tokens.weight
        self.embed_tokens.requires_grad = False
        self.model.embed_tokens.requires_grad = False


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        batch_size, seq_len = input_ids.shape

        visual_embeds = self.vit(pixel_values)
        visual_embeds = self.linear_projector(visual_embeds['last_hidden_state'])

        text_embeds = self.embed_tokens(input_ids)

        text_mask = (input_ids != self.pad_token_id) & (input_ids != self.image_token_id)
        image_mask = input_ids == self.image_token_id

        input_embeds = torch.zeros(batch_size, seq_len, self.config.hidden_size).to('cuda')

        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)

        input_embeds = torch.where(text_mask_expanded, text_embeds, input_embeds).to(torch.bfloat16)

        input_embeds = input_embeds.masked_scatter(image_mask_expanded, visual_embeds)

        return super().forward(None, attention_mask, position_ids, past_key_values, input_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, num_logits_to_keep)


def get_vlm(config):

    vlm_config = VLMConfig()
    processor = VLMProcessor(vlm_config)
    vlm_model = VLMForCausalLM.from_pretrained("google/gemma-2-2b-it", config=vlm_config, torch_dtype=torch.bfloat16,
                                               token=config['token'])
    vlm_model.vit = Dinov2Model.from_ptetrained("facebook/dinov2-base", config=config.vit_config)

    return processor, vlm_model