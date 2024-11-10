from transformers import Gemma2Config, AutoConfig


class VLMConfig(Gemma2Config):
    def __init__(self):
        super().__init__()
        self.visual_embed_dim = 768
        self.pad_token_id = 0
        self.image_token_id = 256000
        self.vit_config = AutoConfig.from_pretrained('facebook/dinov2-base')
        self.num_patches = (self.vit_config.patch_size**2) + 1
        self.text_length = 128
        self.context_length = self.text_length + self.num_patches
