from transformers import Gemma2Config, AutoConfig


class VLMConfig(Gemma2Config):
    def __init__(self, text_length=32, num_patches=64, visual_embed_dim=768, architectures=None):
        pretrained_config = AutoConfig.from_pretrained("google/gemma-2-2b-it")
        super().__init__(**pretrained_config.to_dict())
        self.visual_embed_dim = visual_embed_dim
        self.pad_token_id = 0
        self.image_token_id = 256000
        self.vit_config = AutoConfig.from_pretrained("facebook/dinov2-base")
        self.text_length = text_length
        self.old_num_patches = 257
        self.num_patches = num_patches
        self.context_length = self.text_length + self.num_patches
