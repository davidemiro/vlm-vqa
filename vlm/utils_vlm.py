import torch
from transformers import DefaultDataCollator

class BatchDataCollator(DefaultDataCollator):
    def __call__(self, batch):

        input_ids = []
        attention_mask = []
        labels = []
        input_imgs = []

        for b in batch:
            input_ids.append(torch.LongTensor(b['input_ids']))
            attention_mask.append(torch.LongTensor(b['attention_mask']))
            labels.append(torch.LongTensor(b['labels']))
            input_imgs.append(torch.Tensor(b['pixel_values']))

        return {'input_ids': torch.cat(input_ids,0), 'attention_mask': torch.cat(attention_mask,0), 'labels': torch.cat(labels,0), 'pixel_values': torch.cat(input_imgs,0)}