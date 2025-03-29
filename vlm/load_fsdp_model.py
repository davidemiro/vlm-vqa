
from vlm.modelling_vlm import VLMForCausalLM
from vlm.configuration_vlm import VLMConfig
from torch.distributed.checkpoint import (
    FileSystemReader,
)

import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType
)

import torch.distributed.checkpoint as dist_cp
import os

# Manually set RANK and WORLD_SIZE for a single process
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"  # Required for init
os.environ["MASTER_PORT"] = "12355"

model_name = "mirodavide/vlm-vqa"
model_path = "vlm_checkpoint/"


def load_model_sharded(model, rank, path):

    reader = FileSystemReader(path)

    dist.init_process_group("nccl", rank=rank, world_size=1)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model": model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")


        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
            process_group=dist.group.WORLD,
            coordinator_rank=0,
            no_dist=False,
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
vlm_model.push_to_hub()



