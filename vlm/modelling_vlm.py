from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
from torch import nn
import torch
from transformers import Gemma2Model, PreTrainedModel, HybridCache, GenerationMixin, Cache, StaticCache, Dinov2Model
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from vlm.configuration_vlm import VLMConfig


class VLMForCausalLM(PreTrainedModel):
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_projector = nn.Linear(config.vit_config.visual_embed_dim, config.llm_config.hidden_size, dtype=config.llm_config.torch_dtype)
        self.vit = Dinov2Model.from_pretrained("facebook/dinov2-base", config=config.vit_config, torch_dtype=config.vit_config.torch_dtype)
        self.llm = Gemma2Model.from_pretrained("google/gemma-2-2b-it", config=config.llm_config, torch_dtype=config.llm_config.torch_dtype)
        self.num_patches = config.vit_config.num_patches

        self.image_token_id = self.config.llm_config.image_token_id
        self.pad_token_id = self.config.llm_config.pad_token_id

        old_embed_token = self.llm.embed_tokens
        self.llm.embed_tokens = torch.nn.Embedding(config.llm_config.vocab_size + 1, config.llm_config.hidden_size, self.pad_token_id, dtype=config.llm_config.torch_dtype)
        with torch.no_grad():
            self.llm.embed_tokens.weight[:config.llm_config.vocab_size, :] = old_embed_token.weight

        self.llm.to(device)
        self.vit.to(device)
        self.linear_projector.to(device)

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
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        visual_embeds = self.vit(pixel_values)
        visual_embeds = self.linear_projector(visual_embeds['last_hidden_state'])

        inputs_embeds = self.llm.embed_tokens(input_ids)

        visual_mask = (input_ids == self.config.llm_config.image_token_id)
        visual_mask = visual_mask.unsqueeze(-1)
        visual_mask = visual_mask.repeat(1, 1, self.config.llm_config.hidden_size)
        inputs_embeds = inputs_embeds.masked_scatter(visual_mask, visual_embeds)

        return self.llm(None, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, num_logits_to_keep)


@dataclass
class VLMCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class VLMForConditionalGeneration(VLMForCausalLM, GenerationMixin):

    def __init__(self, config: VLMConfig):
        super().__init__(config, )

    def _update_causal_mask(
        self, attention_mask, token_type_ids, inputs_embeds, past_key_values, cache_position, is_training: bool = False
    ):

        using_static_cache = isinstance(past_key_values, StaticCache)
        dtype = inputs_embeds.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = inputs_embeds.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
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
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
        )
        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_embeds.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
            # we are training thus we need to create a full mask on the image + prefix but causal on suffix
            if is_training:
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )
        return causal_mask

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            pixel_values=None,
            attention_mask=None,
            token_type_ids=None,
            use_cache=True,
            logits_to_keep=None,
            labels=None,
            **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling

        model_inputs = self.llm.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # position_ids in Paligemma are 1-indexed
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1
        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        is_training = token_type_ids is not None and labels is not None
        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
            causal_mask = self._update_causal_mask(
                attention_mask, token_type_ids, past_key_values, cache_position, input_tensor, is_training
            )
            model_inputs["attention_mask"] = causal_mask

        return model_inputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **lm_kwargs,
    ) -> Union[Tuple, VLMCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None

        visual_embeds = self.vit(pixel_values)
        visual_embeds = self.linear_projector(visual_embeds['last_hidden_state'])

        inputs_embeds = self.llm.embed_tokens(input_ids)

        visual_mask = (input_ids == self.config.llm_config.image_token_id)
        visual_mask = visual_mask.unsqueeze(-1)
        visual_mask = visual_mask.repeat(1, 1, self.config.llm_config.hidden_size)

        inputs_embeds = inputs_embeds.masked_scatter(visual_mask, visual_embeds)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1

        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, inputs_embeds, past_key_values, cache_position, is_training
        )

        outputs = self.llm(
            input_ids=None,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **lm_kwargs,
        )

        hidden_states = outputs[0]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.llm.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

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

            flat_logits = shift_logits.view(-1, self.config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        output = VLMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=visual_embeds if pixel_values is not None else None,
        )

        return output if return_dict else output.to_tuple()


