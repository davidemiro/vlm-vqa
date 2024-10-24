from typing import Optional, Union, Tuple, List, Dict, Any

import transformers
from torch import nn
import torch
from transformers import Gemma2ForCausalLM, HybridCache, DataCollatorForLanguageModeling, AutoModel, \
    DataCollatorWithPadding, PreTrainedTokenizer, DefaultDataCollator
from transformers.modeling_outputs import CausalLMOutputWithPast
from image_processing import load_image

import numpy as np

IMAGE_TOKEN = "<image>"


class VLMGemma2ForCausalLM(Gemma2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.linear_projector = nn.Linear(config.visual_embed_dim, config.hidden_size)
        self.vit = AutoModel.from_pretrained('facebook/dinov2-base')
        self.vit.config.image_size

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        text_embeds: Optional[torch.FloatTensor] = None,
        input_imgs: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        visual_embeds = self.vit(input_imgs)
        visual_embeds = self.linear_projector(visual_embeds)
        text_embeds = self.input_embeddings(input_ids)
        input_embeds = torch.cat((visual_embeds, text_embeds), dim=1)

        return super().forward(input_ids, attention_mask, position_ids, past_key_values, input_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, num_logits_to_keep)


class VLMDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizer, context_length, num_patches, img_path, split='train') -> None:
        self.tokenizer = tokenizer
        self.img_path = img_path
        self.split = split
        self.num_patches = num_patches
        self.context_length = context_length

    def __call__(self, row):
        
        text = IMAGE_TOKEN * self.num_patches + "<bos>" + row['question'] + row['answer']
        label = row['answer']

        text_tokenized = self.tokenizer(text, truncation=True)

        label_tokenized = self.tokenizer(label, truncation=True)

        text_tokenized = self.tokenizer.pad(text_tokenized, padding="max_length", max_length=self.context_length)
        label_tokenized = [-100] * (self.context_length - len(label_tokenized)) + label_tokenized

        np_image = load_image(self.img_path, self.split, row['image_id'])

        return {'input_ids': text_tokenized['input_ids'], 'attention_mask': text_tokenized['attention_mask'], 'labels': label_tokenized['input_ids'], 'input_imgs': np_image}


class VLMBatchDataCollator(DefaultDataCollator):
    def __call__(self, batch):

        input_ids = torch.Tensor(batch['input_ids'])
        attention_mask = torch.LongTensor(batch['attention_mask'])
        labels = torch.Tensor(batch['labels'])
        input_imgs = torch.Tensor(np.stack(batch['input_imgs'], axis=0))

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'input_imgs': input_imgs}


def vlm_tokenizer(tokenizer: transformers.GemmaTokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    return tokenizer
















