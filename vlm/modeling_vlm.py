from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
from torch import nn
import torch
from transformers import PreTrainedModel, HybridCache, GenerationMixin, Cache, StaticCache, \
    AutoModelForCausalLM, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from vlm.configuration_vlm import VLMConfig


class VLMForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.linear_projector = nn.Linear(config.vit_config.visual_embed_dim, config.llm_config.hidden_size,dtype=config.llm_config.torch_dtype)
        self.vit = AutoModel.from_pretrained("facebook/dinov2-base", config=config.vit_config, torch_dtype=config.vit_config.torch_dtype, attn_implementation="eager")
        self.llm = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", config=config.llm_config, torch_dtype=config.llm_config.torch_dtype, attn_implementation="eager")
        self.num_patches = config.vit_config.num_patches

        self.image_token_id = self.config.llm_config.image_token_id
        self.pad_token_id = self.config.llm_config.pad_token_id

        old_embed_token = self.llm.model.embed_tokens
        self.llm.model.embed_tokens = torch.nn.Embedding(config.llm_config.vocab_size + 1, config.llm_config.hidden_size, self.pad_token_id, dtype=config.llm_config.torch_dtype)
        with torch.no_grad():
            self.llm.model.embed_tokens.weight[:config.llm_config.vocab_size, :] = old_embed_token.weight
        del old_embed_token

        self.llm.model.embed_tokens.weight.requires_grad = False


    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids=None,
        past_key_values=None,
        cache_position=None,
        input_tensor=None,
        is_training: Optional[bool] = None,
    ):
        if self.config.llm_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        is_training = is_training if is_training is not None else self.training
        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        if input_tensor is None:
            input_tensor = attention_mask

        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device
        )
        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]

            # First unmask prefix tokens during training
            if is_training:
                if token_type_ids is None:
                    raise ValueError("Token type ids must be provided during training")
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )

            # Then apply padding mask (will mask pad tokens)
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        return causal_mask



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
        token_type_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        is_training = token_type_ids is not None and labels is not None

        visual_embeds = self.vit(pixel_values)
        visual_embeds = self.linear_projector(visual_embeds['last_hidden_state'])

        inputs_embeds = self.llm.model.embed_tokens(input_ids)

        visual_mask = (input_ids == self.config.llm_config.image_token_id)
        visual_mask = visual_mask.unsqueeze(-1)
        visual_mask = visual_mask.repeat(1, 1, self.config.llm_config.hidden_size)
        inputs_embeds = inputs_embeds.masked_scatter(visual_mask, visual_embeds)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        del visual_mask
        del visual_embeds
        del input_ids

        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
        )

        outputs = self.llm(None,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        inputs_embeds,
                        labels,
                        use_cache,
                        output_attentions,
                        output_hidden_states,
                        return_dict,
                        cache_position,
                        num_logits_to_keep)

        logits = outputs.logits

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1]:].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.llm_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        outputs.loss = loss

        return outputs


@dataclass
class VLMCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


