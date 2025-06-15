import torch
from transformers import AutoConfig, PretrainedConfig

IMAGE_TOKEN = "<image>"


class VLMConfig(PretrainedConfig):
    def __init__(self, text_length=32, num_patches=64, visual_embed_dim=768, torch_dtype=torch.float16, dropout=0.1, **kwargs):
        self.llm_config = AutoConfig.from_pretrained("google/gemma-2-2b-it")
        self.vit_config = AutoConfig.from_pretrained("facebook/dinov2-base")

        self.vit_config.visual_embed_dim = visual_embed_dim
        self.vit_config.old_num_patches = 257
        self.vit_config.num_patches = num_patches
        self.vit_config.torch_dtype = torch_dtype
        self.vit_config.hidden_dropout_prob = dropout
        self.vit_config.attention_probs_dropout_prob = dropout

        self.llm_config.pad_token_id = 0
        self.llm_config.image_token = IMAGE_TOKEN
        self.llm_config.image_token_id = 256000
        self.llm_config.text_length = text_length
        self.llm_config.context_length = self.llm_config.text_length + self.vit_config.num_patches
        self.llm_config.torch_dtype = torch_dtype
        self.llm_config.attention_dropout = dropout

        super(VLMConfig, self).__init__(**kwargs)
