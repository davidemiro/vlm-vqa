from transformers import AutoConfig, PretrainedConfig

IMAGE_TOKEN = "<image>"

class VLMConfig(PretrainedConfig):
    def __init__(self, text_length=32, num_patches=64, visual_embed_dim=768, **kwargs):
        self.lm_config = AutoConfig.from_pretrained("google/gemma-2-2b-it")
        self.vit_config = AutoConfig.from_pretrained("facebook/dinov2-base")

        self.vit_config.visual_embed_dim = visual_embed_dim
        self.vit_config.old_num_patches = 257
        self.vit_config.num_patches = num_patches

        self.lm_config.vocab_size += 1
        self.lm_config.pad_token_id = 0
        self.lm_config.image_token = IMAGE_TOKEN
        self.lm_config.image_token_id = 256000
        self.lm_config.text_length = text_length
        self.lm_config.context_length = self.lm_config.text_length + self.num_patches


        super(VLMConfig, self).__init__(**kwargs)
