
from vlm.modelling_vlm import VLMForCausalLM
from vlm.configuration_vlm import VLMConfig
from torch.distributed.checkpoint import (
    FileSystemReader,
)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType
)

import torch.distributed.checkpoint as dist_cp

model_name = "mirodavide/vlm-vqa"
model_path = "vlm_checkpoint/"



def from_fsdp_shards_to_hf(model, model_path):

    state_dict = {
        "model": model.state_dict()
    }

    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=FileSystemReader(model_path),
        no_dist=True,
    )

    model.load_state_dict(state_dict["model"])

    print(f"Sharded state checkpoint loaded from {model_path}")
    return model


def load_model_sharded(model, rank, path):

    reader = FileSystemReader(path)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model": model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")

        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            print(f"checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        model.load_state_dict(checkpoint["model"])
    if rank == 0:
        print(f"Sharded state checkpoint loaded from {path}")

        return model


vlm_config = VLMConfig(text_length=32, num_patches=257, visual_embed_dim=768)
vlm_model = VLMForCausalLM(vlm_config)

load_model_sharded(vlm_model, 4, model_path)

vlm_model.save_pretrained("vlm-vqa-1.0")



