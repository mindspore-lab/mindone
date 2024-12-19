import importlib
import os
from typing import List

import cv2
import numpy as np

from mindspore import nn


def _get_subcell(mod: nn.Cell, target: str) -> "nn.Cell":
    """See https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule"""
    if target == "":
        return mod

    atoms: List[str] = target.split(".")
    for item in atoms:
        if not hasattr(mod, item):
            raise AttributeError(f"{mod.__class__.__name__} has no attribute `{item}`")
        mod = getattr(mod, item)
        if not isinstance(mod, nn.Cell):
            raise AttributeError(f"`{item}` is not an nn.Cell")
    return mod


def _get_submodules(model, key):
    parent = _get_subcell(model, ".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = _get_subcell(model, key)
    return parent, target, target_name


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def freeze_params(m: nn.Cell):
    for p in m.get_parameters():
        p.requires_grad = False


def check_istarget(name, para_list):
    """
    name: full name of source para
    para_list: partial name of target para
    """
    istarget = False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))["arr_0"] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)["arr_0"] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image
