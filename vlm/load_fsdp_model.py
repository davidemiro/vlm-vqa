from vlm.configuration_vlm import VLMConfig
from vlm.modeling_vlm import VLMForCausalLM
import torch




def load_model(fsdp_model_path):

    vlm_config = VLMConfig(text_length=64, num_patches=257, visual_embed_dim=768)
    base_model = VLMForCausalLM(vlm_config)

    base_state_dict = base_model.state_dict()
    load_state_dict = torch.load(fsdp_model_path, map_location="cpu")["model"]

    load_state_dict = {k.replace("_orig_mod.", ""): v for k, v in load_state_dict.items()}

    for k in base_state_dict.keys():
        if base_state_dict[k].shape != load_state_dict[k].shape:
            print("{} {} {}".format(k, base_state_dict[k].shape, load_state_dict[k].shape))
            try:
                load_state_dict[k] = load_state_dict[k].reshape(base_state_dict[k].shape)
            except:
                load_state_dict[k] = base_state_dict[k]

    base_model.load_state_dict(load_state_dict)

    return base_model







