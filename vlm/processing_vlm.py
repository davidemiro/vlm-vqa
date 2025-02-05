from transformers import AutoProcessor, AutoImageProcessor, AutoTokenizer
from vlm.configuration_vlm import VLMConfig
import torch

class VLMProcessor(AutoProcessor):
    def __init__(self, config: VLMConfig, token: str) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it', token=token)
        self.image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', token=token)
        self.context_length = config.context_length

    def _training_processing(self, text, image, label, return_tensors="pt"):
        text = "<bos>" + text + label
        text_tokenized = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.context_length,
                                        return_tensors=return_tensors)

        label_tokenized = self.tokenizer(label, truncation=True, padding="max_length", max_length=self.context_length,
                                         return_tensors=return_tensors)['input_ids']
        label_tokenized[label_tokenized == 0] = -100

        pixel_values = self.image_processor(images=image, do_resize=True, size={"height": 224, "width": 224}, return_tensors=return_tensors)['pixel_values']

        print(pixel_values.shape)
        return {'input_ids': text_tokenized['input_ids'], 'attention_mask': text_tokenized['attention_mask'],
                'labels': label_tokenized, 'pixel_values': pixel_values}

    def _inference_processing(self, text, image, return_tensors="np"):
        text = "<bos>" + text
        text_tokenized = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.context_length,
                                        return_tensors=return_tensors)

        pixel_values = self.image_processor(images=image, padding=True, return_tensors=return_tensors)['pixel_values']

        return {'input_ids': text_tokenized['input_ids'], 'attention_mask': text_tokenized['attention_mask'],
                'pixel_values': pixel_values}

    def __call__(self, text, image, label=None, return_tensors="pt"):
        if label is None:
            return self._inference_processing(text, image, return_tensors=return_tensors)
        else:
            return self._training_processing(text, image, label, return_tensors=return_tensors)



