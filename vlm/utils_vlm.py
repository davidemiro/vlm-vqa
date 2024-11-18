import torch
from transformers import DefaultDataCollator
from PIL import Image
from vlm.processing_vlm import VLMProcessor
import os


class BatchDataCollator(DefaultDataCollator):

    def __init__(self, processor: VLMProcessor) -> None:
        self.processor = processor

    def __call__(self, batch):

        input_ids = []
        attention_masks = []
        labels = []
        pixel_values = []

        for row in batch:
            image = self._load_image(row['img_path'], row['split'], row['image_id'])
            row_dict = self.processor(text=row['question'], image=image, label=row['answer'], return_tensors="pt")

            input_ids.append(row_dict['input_ids'])
            attention_masks.append(row_dict['attention_mask'])
            labels.append(row_dict['labels'])
            pixel_values.append(row_dict['pixel_values'])

        return {'input_ids': torch.cat(input_ids,0), 'attention_mask': torch.cat(attention_masks,0), 'labels': torch.cat(labels,0), 'pixel_values': torch.cat(pixel_values,0)}

    def _load_image(self, path, split, image_id):
        image_id = "0" * (12 - len(str(image_id))) + str(image_id)
        img = Image.open(os.path.join(path, "COCO_{}2014_{}.jpg".format(split, image_id)))
        return img