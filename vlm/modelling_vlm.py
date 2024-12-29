from typing import Optional, Union, Tuple
from torch import nn
import torch
from transformers import Gemma2ForCausalLM, HybridCache, Dinov2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from vlm.configuration_vlm import VLMConfig
from vlm.processing_vlm import VLMProcessor


class VLMForCausalLM(Gemma2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.linear_projector = nn.Linear(config.visual_embed_dim, config.hidden_size)
        self.linear_projector_visual_embedding = nn.Linear(config.num_patches, 64)
        self.vit = Dinov2Model(config=config.vit_config)
        self.num_patches = 64

        self.image_token_id = self.config.image_token_id
        self.pad_token_id = self.config.pad_token_id

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

        visual_embeds = self.vit(pixel_values)
        visual_embeds = self.linear_projector(visual_embeds['last_hidden_state'])
        print(visual_embeds.shape)
        visual_embeds = visual_embeds.permute(0, 2, 1) #[batch_size, hidden_size, num_patches]
        print(visual_embeds.shape)
        visual_embeds = self.linear_projector_visual_embedding(visual_embeds) #[batch_size, hidden_size, new_num_patches]
        print(visual_embeds.shape)
        visual_embeds = visual_embeds.permute(0, 2, 1) #[batch_size, hidden_size, new_num_patches]
        print(visual_embeds.shape)

        print(input_ids.shape)
        input_ids = input_ids[:, self.num_patches:]
        print(input_ids.shape)

        text_embeds = self.model.embed_tokens(input_ids)
        print(text_embeds.shape)

        input_embeds = torch.cat((text_embeds, visual_embeds), dim=1)
        print(input_embeds.shape)

        return super().forward(None, attention_mask, position_ids, past_key_values, input_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, num_logits_to_keep)


def get_vlm(config):

    vlm_config = VLMConfig(text_length=int(config["text_length"]), visual_length=int(config["visual_length"]), visual_embed_dim=int(config["visual_embed_dim"]))
    processor = VLMProcessor(vlm_config)
    vlm_model = VLMForCausalLM.from_pretrained("google/gemma-2-2b-it", config=vlm_config, torch_dtype=torch.bfloat16,
                                               token=config['token'])
    vlm_model.vit = Dinov2Model.from_pretrained("facebook/dinov2-base", config=vlm_config.vit_config, torch_dtype=torch.bfloat16)

    return processor, vlm_model