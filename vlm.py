import transformers
from torch import nn
import torch

IMAGE_TOKEN = "<image>"


def vlm_tokenizer(tokenizer: transformers.GemmaTokenizer):

    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    return tokenizer


class VLMEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, padding_idx, visual_embed_dim):
        super(VLMEmbedding, self).__init__()
        self.linear_projector = nn.Linear(visual_embed_dim, hidden_size)
        self.text_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

    def forward(self, text_tokens, visual_embeddings):

        visual_embeddings = self.linear_projector(visual_embeddings)
        text_embeddings = self.text_embeddings(text_tokens)
        return torch.cat((visual_embeddings, text_embeddings), dim=1)









