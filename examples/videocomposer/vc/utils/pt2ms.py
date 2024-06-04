import difflib
import logging
import os
from copy import deepcopy

import numpy as np

import mindspore as ms
from mindspore import nn

__all__ = [
    "get_pt2ms_mappings",
    "convert_state_dict",
    "load_pt_weights_in_model",
]

_logger = logging.getLogger(__name__)


def get_pt2ms_mappings(model: nn.Cell):
    mappings = {}  # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)

    def check_key(k):
        if k in mappings:
            raise KeyError(f"param name {k} is already in mapping!")

    for name, cell in model.cells_and_names():
        if isinstance(cell, nn.Conv1d):
            check_key(f"{name}.weight")
            mappings[f"{name}.weight"] = f"{name}.weight", lambda x: np.expand_dims(x, axis=-2)
        elif isinstance(cell, nn.Embedding):
            check_key(f"{name}.weight")
            mappings[f"{name}.weight"] = f"{name}.embedding_table", lambda x: x
        elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            check_key(f"{name}.weight")
            mappings[f"{name}.weight"] = f"{name}.gamma", lambda x: x
            check_key(f"{name}.bias")
            mappings[f"{name}.bias"] = f"{name}.beta", lambda x: x
    return mappings


def convert_state_dict(model, state_dict_pt):
    mappings = get_pt2ms_mappings(model)
    state_dict_ms = {}
    for name_pt, data_pt in state_dict_pt.items():
        name_ms, data_mapping = mappings.get(name_pt, (name_pt, lambda x: x))
        data_ms = data_mapping(data_pt)
        state_dict_ms[name_ms] = ms.Parameter(data_ms.astype(np.float32), name=name_ms)
    return state_dict_ms


def auto_map(model, param_dict, verbose=True):
    """Raname part of the param_dict such that names from checkpoint and model are consistent"""
    updated_param_dict = deepcopy(param_dict)
    net_param = model.get_parameters()
    ckpt_param = list(updated_param_dict.keys())
    remap = {}
    for param in net_param:
        if param.name not in ckpt_param:
            poss = difflib.get_close_matches(param.name, ckpt_param, n=3, cutoff=0.6)
            if len(poss) > 0:
                if verbose:
                    _logger.info(f"{param.name} not exist in checkpoint. Find a most matched one: {poss[0]}")
                updated_param_dict[param.name] = updated_param_dict.pop(poss[0])  # replace
                remap[param.name] = poss[0]
            else:
                raise ValueError(f"Cannot find any matching param with {param.name} from: ", ckpt_param)

    if remap != {}:
        _logger.warning("Auto mapping succeed. Please check the found mapping names to ensure correctness")
        _logger.info("\tNet Param\t<---\tCkpt Param")
        for k in remap:
            _logger.info(f"\t{k}\t<---\t{remap[k]}")
    return updated_param_dict


def load_pt_weights_in_model(model, checkpoint_file_pt, state_dict_refiners=None):
    checkpoint_file_ms = f"{os.path.splitext(checkpoint_file_pt)[0]}.ckpt"
    if not os.path.exists(checkpoint_file_ms):  # try to load weights from intermediary numpy file.
        checkpoint_file_np = f"{os.path.splitext(checkpoint_file_pt)[0]}.npy"
        if not os.path.exists(checkpoint_file_np):
            raise FileNotFoundError(f"You need to manually convert {checkpoint_file_pt} to {checkpoint_file_np}")
        sd_original = np.load(checkpoint_file_np, allow_pickle=True).item()
        # refine state dict of pytorch
        sd_refined = sd_original
        if state_dict_refiners:
            for refine_fn in state_dict_refiners:
                sd_refined = refine_fn(sd_refined)
        # convert state_dict from pytorch to mindspore
        sd = convert_state_dict(model, sd_refined)
        # save converted state_dict as cache
        ms.save_checkpoint([{"name": k, "data": v} for k, v in sd.items()], checkpoint_file_ms)
    else:  # directly load weights from cached mindspore file.
        sd = ms.load_checkpoint(checkpoint_file_ms)
    sd = auto_map(model, sd)  # automatically map the ms parameter names with the key names in the checkpoint file.
    param_not_load, ckpt_not_load = ms.load_param_into_net(model, sd, strict_load=True)
    if param_not_load:
        _logger.warning(f"{param_not_load} in network is not loaded")
    if ckpt_not_load:
        _logger.warning(f"{ckpt_not_load} in checkpoint is not loaded!")


if __name__ == "__main__":
    #############################################################################
    # Script for converting pytorch serialized file to intermediary numpy file. #
    #############################################################################
    import shutil

    import torch

    pytorch_model_paths = [
        "model_weights/non_ema_228000.pth",
        "model_weights/v2-1_512-ema-pruned.ckpt",
        "model_weights/sketch_simplification_gan.pth",
        "model_weights/table5_pidinet.pth",
    ]

    for pytorch_model_path in pytorch_model_paths:
        print("-" * 80)
        print(f"Converting {pytorch_model_path}")
        print("-" * 80)
        state_dict = torch.load(pytorch_model_path, map_location="cpu")
        if "state_dict" in state_dict:
            print(f"Popping 'state_dict' from {state_dict.keys()}")
            state_dict = state_dict["state_dict"]
        print(state_dict.keys())
        state_dict_np = {k: v.numpy() for k, v in state_dict.items()}
        np.save(f"{os.path.splitext(pytorch_model_path)[0]}.npy", state_dict_np)

    print("[WARNING] Backing up original pytorch file to avoid potential filename conflict.")
    for src_path in pytorch_model_paths:
        dst_path = f"{src_path}.bak"
        print(f"moving {src_path} to {dst_path} ...")
        shutil.move(src_path, dst_path)
