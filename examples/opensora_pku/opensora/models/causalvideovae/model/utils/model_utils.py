import importlib

import numpy as np

import mindspore as ms

Module = str
MODULES_BASE = "opensora.models.causalvideovae.model.modules."


def resolve_str_to_obj(str_val, append=True):
    if append:
        str_val = MODULES_BASE + str_val
    module_name, class_name = str_val.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_obj_from_str(string: str, reload: bool = False) -> object:
    """TODO: debug
    if string.startswith('mindone'):
        string = '../../' + string
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_torch_state_dict_to_ms_ckpt(ckpt_file):
    import torch

    source_data = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    if "state_dict" in source_data:
        source_data = source_data["state_dict"]
    if "ema" in source_data:
        source_data = source_data["ema"]

    target_data = {}
    for k in source_data:
        val = source_data[k].detach().numpy().astype(np.float32)
        # print(type(val), val.dtype, val.shape)
        target_data.update({k: ms.Parameter(ms.Tensor(val, dtype=ms.float32))})
    return target_data
