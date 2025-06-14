import torch
from transformers import DefaultDataCollator, Dinov2Model
from PIL import Image
import os

from vlm.configuration_vlm import VLMConfig
from vlm.processing_vlm import VLMProcessor
from vlm.modeling_vlm import VLMForCausalLM


class BatchDataCollator(DefaultDataCollator):

    def __init__(self, processor: VLMProcessor) -> None:
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, batch):

        input_ids = []
        attention_masks = []
        labels = []
        pixel_values = []
        token_type_ids = []

        for row in batch:
            image = self._load_image(row['img_path'], row['split'], row['image_id'])
            row_dict = self.processor(text=row['question'], image=image, label=row['answer'], return_tensors="pt")

            input_ids.append(row_dict['input_ids'])
            attention_masks.append(row_dict['attention_mask'])
            labels.append(row_dict['labels'])
            pixel_values.append(row_dict['pixel_values'])
            token_type_ids.append(row_dict['token_type_ids'])


        return {
            'input_ids': torch.cat(input_ids, 0),
            'attention_mask': torch.cat(attention_masks, 0),
            'labels': torch.cat(labels, 0),
            'token_type_ids': torch.cat(token_type_ids, 0),
            'pixel_values': torch.cat(pixel_values, 0),

        }

    def _load_image(self, path, split, image_id):
        image_id = "0" * (12 - len(str(image_id))) + str(image_id)
        img = Image.open(os.path.join(path, "COCO_{}2014_{}.jpg".format(split, image_id)))
        return img


def get_vlm(config):

    vlm_config = VLMConfig(text_length=int(config["text_length"]), num_patches=int(config["num_patches"]), visual_embed_dim=int(config["visual_embed_dim"]),torch_dtype=torch.float16, dropout=float(config["dropout"]))
    processor = VLMProcessor(vlm_config, config['token'])
    vlm_model = VLMForCausalLM(config=vlm_config)

    return processor, vlm_model, vlm_config
