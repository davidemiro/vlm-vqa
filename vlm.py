from typing import Optional, Union, Tuple

import transformers
from torch import nn
import torch
from transformers import Gemma2ForCausalLM, HybridCache, AutoModel, PreTrainedTokenizer, DefaultDataCollator
from transformers.modeling_outputs import CausalLMOutputWithPast
from PIL import Image
import os
import numpy as np


IMAGE_TOKEN = "<image>"


class VLMGemma2ForCausalLM(Gemma2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.linear_projector = nn.Linear(config.visual_embed_dim, config.hidden_size)
        self.vit = AutoModel.from_pretrained(config.vit_name)

        self.image_token_id = self.config.image_token_id
        self.pad_token_id = self.config.pad_token_id

        self.embed_tokens = nn.Embedding(self.config.vocab_size + 1, self.config.hidden_size)

        with torch.no_grad():
            self.embed_tokens.weight[:self.config.vocab_size, :] = self.model.embed_tokens.weight

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

        batch_size, seq_len = input_ids.shape

        visual_embeds = self.vit(input_imgs)
        visual_embeds = self.linear_projector(visual_embeds['last_hidden_state'])

        text_embeds = self.embed_tokens(input_ids)

        text_mask = (input_ids != self.pad_token_id) & (input_ids != self.image_token_id)
        image_mask = input_ids == self.image_token_id

        input_embeds = torch.zeros(batch_size, seq_len, self.config.hidden_size)

        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)

        input_embeds = torch.where(text_mask_expanded, text_embeds, input_embeds)
        input_embeds = input_embeds.masked_scatter(image_mask_expanded, visual_embeds)

        return super().forward(None, attention_mask, position_ids, past_key_values, input_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, num_logits_to_keep)


class VLMDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizer, text_length, processor, num_patches, img_path, split='train') -> None:
        self.tokenizer = tokenizer
        self.img_path = img_path
        self.split = split
        self.num_patches = num_patches
        self.text_length = text_length
        self.context_length = self.num_patches + self.text_length
        self.processor = processor

    def _load_image(self, path, split, image_id):
        image_id = "0" * (12 - len(str(image_id))) + str(image_id)
        img = Image.open(os.path.join(path, "COCO_{}2014_{}.jpg".format(split, image_id)))
        return img

    def __call__(self, row):
        
        text = IMAGE_TOKEN * self.num_patches + "<bos>" + row['question'] + row['answer']
        label = row['answer']

        text_tokenized = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.context_length, return_tensors="np")

        label_tokenized = self.tokenizer(label, truncation=True, padding="max_length", max_length=self.context_length, return_tensors="np")['input_ids']
        label_tokenized[label_tokenized == 0] = -100

        img = self._load_image(self.img_path, self.split, row['image_id'])
        pt_image = self.processor(images=img, return_tensors="np")['pixel_values']

        return {'input_ids': text_tokenized['input_ids'], 'attention_mask': text_tokenized['attention_mask'], 'labels': label_tokenized, 'input_imgs': pt_image}


class VLMBatchDataCollator(DefaultDataCollator):
    def __call__(self, batch):

        input_ids = []
        attention_mask = []
        labels = []
        input_imgs = []

        for b in batch:
            input_ids.append(torch.LongTensor(b['input_ids']))
            attention_mask.append(torch.LongTensor(b['attention_mask']))
            labels.append(torch.LongTensor(b['labels']))
            input_imgs.append(torch.Tensor(b['input_imgs']))

        return {'input_ids': torch.cat(input_ids,0), 'attention_mask': torch.cat(attention_mask,0), 'labels': torch.cat(labels,0), 'input_imgs': torch.cat(input_imgs,0)}


def vlm_tokenizer(tokenizer: transformers.GemmaTokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    return tokenizer


















