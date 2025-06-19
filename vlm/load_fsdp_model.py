import pickle
from typing import Any, Dict

import torch
from torch.distributed.checkpoint.planner_helpers import _create_read_items
from torch.distributed.tensor import DTensor

from vlm.modeling_vlm import VLMForCausalLM
from vlm.configuration_vlm import VLMConfig


import torch.distributed.checkpoint as dist_cp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

from torch.distributed.checkpoint import (
    FileSystemReader,
    DefaultLoadPlanner, ReadItem, LoadPlan, Metadata, TensorStorageMetadata
)

class LoadVLMPlanner(DefaultLoadPlanner):

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        self.state_dict[read_item.dest_index.fqn] = tensor



    def create_default_local_load_plan(self,
            state_dict: Dict[str, Any], metadata: Metadata, strict: bool = True
    ) -> LoadPlan:
        requests = []
        """
        Create the ``LoadPlan`` used by DefaultLoadPlanner.

        It produces one read item per value in ``state_dict`` using the metadata in ``metadata``.

        The default behavior is to match key exactly between state_dict and metadata.
        It handles resharding by issuing multiple read requests against storage in order to match
        load requirements.
        """

        for fqn, obj in state_dict.items():

            fqn = fqn.replace("model","model._orig_mod")
            if fqn not in metadata.state_dict_metadata:
                if strict:
                    raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
                else:
                    continue

            md = metadata.state_dict_metadata[fqn]
            if (
                    isinstance(md, TensorStorageMetadata)
                    and getattr(obj, "size", None) is not None
                    and md.size != obj.size()
            ):
                raise ValueError(
                    f"Size mismatch between saved {md.size} and current: {obj.size()} for {fqn}",
                )
            # Since DTensor supports submesh, adding extra check to ensure _create_read_items()
            # gets called only when the current rank is part of the mesh for the corresponding DTensor.
            if isinstance(obj, DTensor):
                if obj.device_mesh.get_coordinate() is not None:
                    requests += _create_read_items(fqn, md, obj)
            else:
                requests += _create_read_items(fqn, md, obj)

        return LoadPlan(requests)

    def create_local_plan(self) -> LoadPlan:
        assert self.metadata is not None

        return self.create_default_local_load_plan(
            self.state_dict, self.metadata, True
        )

def load_model_sharded(model, path):



    reader = FileSystemReader(path)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model": model.state_dict()}
        metadata = None
        with open(path+"/.metadata", "rb") as f:
            metadata = pickle.load(f)

        dist_cp.load(
            state_dict=checkpoint,
            storage_reader=reader
        )
        model.load_state_dict(checkpoint["model"])


vlm_config = VLMConfig(text_length=64, num_patches=257, visual_embed_dim=768)
vlm_model = VLMForCausalLM(vlm_config)

load_model_sharded(vlm_model, "/Users/davidemiro/vlm-gemma-2-2b/vlm-vqa/vlm-vqa/checkpoint-92/pytorch_model_fsdp_0")

vlm_model.save_pretrained("/Users/davidemiro/vlm-gemma-2-2b/vlm-vqa/vlm-vqa/vlm-vqa-1.0")




