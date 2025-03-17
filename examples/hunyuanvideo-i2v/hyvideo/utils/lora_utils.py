import mindspore as ms
from mindspore import mint


# load kohya lora for diffusers pipeline
def load_lora_for_pipeline(
    pipeline,
    lora_path,
    LORA_PREFIX_TRANSFORMER="",
    LORA_PREFIX_TEXT_ENCODER="",
    alpha=1.0,
):
    # load LoRA weight from .safetensors
    state_dict = ms.load_checkpoint(lora_path, format="safetensors")

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
            curr_layer = pipeline.transformer

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(ms.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(ms.float32)
            ori_param = curr_layer.weight.value()
            curr_layer.weight.set_data(ori_param + alpha * mint.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3))
        else:
            weight_up = state_dict[pair_keys[0]].to(ms.float32)
            weight_down = state_dict[pair_keys[1]].to(ms.float32)
            ori_param = curr_layer.weight.value()
            curr_layer.weight.set_data(ori_param + alpha * mint.mm(weight_up, weight_down))

        # update visited list
        for item in pair_keys:
            visited.append(item)
    del state_dict

    return pipeline
