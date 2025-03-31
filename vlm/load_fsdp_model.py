from vlm.modelling_vlm import VLMForCausalLM
from vlm.configuration_vlm import VLMConfig
import os


import torch.distributed.checkpoint as dist_cp



model_name = "mirodavide/vlm-vqa"
model_path = "/content/drive/MyDrive/train_vlm_vqa_1e05_lr"


def load_model_sharded(model, model_path):
    print("E")
    state_dict = {
        "model": model.state_dict(),
    }

    # since no progress group is initialized, DCP will disable any collectives.
    dist_cp.load(
        state_dict=state_dict,
        checkpoint_id=model_path,
    )
    print("F")
    model.load_state_dict(state_dict["model"])
    print("G")

    return model

print("A")
vlm_config = VLMConfig(text_length=32, num_patches=257, visual_embed_dim=768)
print("B")
vlm_model = VLMForCausalLM(vlm_config)
print("C")
load_model_sharded(vlm_model, model_path)
print("D")
vlm_model.save_pretrained("vlm-vqa-1.0")
vlm_model.push_to_hub()



