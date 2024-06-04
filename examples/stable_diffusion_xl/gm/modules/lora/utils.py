# reference to https://github.com/microsoft/LoRA

from typing import Dict

from gm.modules.lora.layers import LoRALayer

import mindspore as ms
from mindspore import Tensor, nn


def mark_only_lora_as_trainable(model: nn.Cell, bias: str = "none") -> None:
    for n, p in model.parameters_and_names():
        if "lora_" not in n:
            p.requires_grad = False

    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.parameters_and_names():
            if "bias" in n:
                p.requires_grad = True
            if "beta" in n:  # bn/ln bias
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.cells():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
            if isinstance(m, LoRALayer) and hasattr(m, "beta") and m.beta is not None:
                m.beta.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Cell, bias: str = "none") -> Dict[str, Tensor]:
    my_state_dict = {n: p for n, p in model.parameters_and_names()}

    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def lora_save_checkpoint(state_dict, ckpt_path):
    param_list = [{"name": n, "data": p} for n, p in state_dict]
    ms.save_checkpoint(param_list, ckpt_path)
