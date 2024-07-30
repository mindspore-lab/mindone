import os
import sys
from pathlib import Path

import numpy as np
from jsonargparse.typing import Path_fr, path_type
from omegaconf import OmegaConf

from mindspore import Parameter, load_param_into_net, save_checkpoint

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from sgm.helpers import create_model_sv3d as create_model

Path_dcc = path_type("dcc")  # path to a directory that can be created if it does not exist


import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import HfApi, hf_hub_download

# from huggingface_hub.file_download import repo_folder_name
from safetensors.torch import _find_shared_tensors, _is_complete, load_file, save_file


def _remove_duplicate_names(
    state_dict: Dict[str, torch.Tensor],
    *,
    preferred_names: List[str] = None,
    discard_names: List[str] = None,
) -> Dict[str, List[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set([name for name in shared if _is_complete(state_dict[name])])
        if not complete_names:
            if len(shared) == 1:
                # Force contiguous
                name = list(shared)[0]
                state_dict[name] = state_dict[name].clone()
                complete_names = {name}
            else:
                raise RuntimeError(
                    f"Error while trying to find names to remove to save state dict, but found no suitable name to keep for saving amongst: {shared}."
                )

        keep_name = sorted(list(complete_names))[0]

        # Mecanism to preferentially select keys to keep
        # coming from the on-disk file to allow
        # loading models saved with a different choice
        # of keep_name
        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove


def get_discard_names(model_id: str, revision: Optional[str], folder: str, token: Optional[str]) -> List[str]:
    try:
        import json

        import transformers

        config_filename = hf_hub_download(
            model_id, revision=revision, filename="config.json", token=token, cache_dir=folder
        )
        with open(config_filename, "r") as f:
            config = json.load(f)
        architecture = config["architectures"][0]

        class_ = getattr(transformers, architecture)

        # Name for this varible depends on transformers version.
        discard_names = getattr(class_, "_tied_weights_keys", [])

    except Exception:
        discard_names = []
    return discard_names


class AlreadyExists(Exception):
    pass


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError()


def rename(pt_filename: str) -> str:
    filename, ext = os.path.splitext(pt_filename)
    local = f"{filename}.safetensors"
    local = local.replace("pytorch_model", "model")
    return local


def convert_single(
    model_id: str, *, revision: Optional[str], folder: str, token: Optional[str], discard_names: List[str]
):
    pt_filename = hf_hub_download(
        repo_id=model_id, revision=revision, filename="pytorch_model.bin", token=token, cache_dir=folder
    )

    sf_name = "model.safetensors"
    sf_filename = os.path.join(folder, sf_name)
    convert_file(pt_filename, sf_filename, discard_names)
    errors: List[Tuple[str, "Exception"]] = []
    return errors


def convert_file(
    pt_filename: str,
    sf_filename: str,
    discard_names: List[str],
):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    to_removes = _remove_duplicate_names(loaded, discard_names=discard_names)

    metadata = {"format": "pt"}
    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del loaded[to_remove]
    # Force tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata=metadata)
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def convert_bin2safetensors(
    api: "HfApi", model_id: str, revision: Optional[str] = None, force: bool = False
) -> List[Tuple[str, "Exception"]]:
    filenames = set("pytorch_model.bin")
    # Uncomment this if you have hf access smoothly
    # info = api.model_info(model_id, revision=revision)
    # filenames = set(s.rfilename for s in info.siblings)
    folder = "./ckpts"
    os.makedirs(folder, exist_ok=True)
    print(f"current folder is {folder}")

    library_name = "transformers"
    if any(filename.endswith(".safetensors") for filename in filenames) and not force:
        raise AlreadyExists(f"Model {model_id} is already converted, skipping..")
    elif library_name == "transformers":
        discard_names = get_discard_names(model_id, revision=revision, folder=folder, token=api.token)
        if "pytorch_model.bin" in filenames:
            errors = convert_single(
                model_id, revision=revision, folder=folder, token=api.token, discard_names=discard_names
            )
        else:
            raise RuntimeError(f"Model {model_id} doesn't seem to be a valid pytorch model. Cannot convert")
    return errors


def convert_torch2ms(pt_weights_file: Path_fr, config: Path_fr, out_dir: Optional[Path_dcc] = None):
    """
    Convert PyTorch weights to MindSpore format.

    Args:
        pt_weights_file: Path to the PyTorch weights file.
        config: Path to the VideoLDM config file.
        out_dir: (optional) Path to directory where the converted checkpoint will be saved.
                If not provided, the converted checkpoint will be saved in the same directory as the PyTorch checkpoint.
    Raises:
        ValueError: If some parameters were not loaded during the conversion process.
    """
    pt_weights_file = Path(pt_weights_file)
    pt_ckpt = load_file(pt_weights_file)
    pt_keys = list(pt_ckpt.keys())

    config = OmegaConf.load(config.absolute)
    network, _ = create_model(config, freeze=True)
    ms_weights = network.parameters_dict()
    ms_keys = list(ms_weights.keys())

    # after sorting, PyTorch and MindSpore parameters are aligned
    pt_keys = sorted(pt_keys)
    ms_keys = sorted(ms_keys)

    for pt_key, ms_key in zip(pt_keys, ms_keys):
        ms_weights[ms_key] = Parameter(np.array(pt_ckpt[pt_key]), name=ms_key)

    param_not_load, _ = load_param_into_net(network, ms_weights)
    if param_not_load:
        raise ValueError(f"Something went wrong. The following parameters were not loaded: \n{param_not_load}")

    out_dir = pt_weights_file.parent / "ms_models" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir / (pt_weights_file.stem + ".ckpt")

    save_checkpoint(network, str(out_dir))
    print(f"Conversion completed. Checkpoint is saved to: \n{out_dir}")


if __name__ == "__main__":
    DESCRIPTION = """
    Simple utility tool to convert automatically some weights on the hub to `safetensors` format.
    It is PyTorch exclusive for now.
    It works by downloading the weights (PT), converting them locally, and uploading them back
    as a PR on the hub.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--model_id",
        default="facebook/dino-vitb16",
        type=str,
        help="The name of the model on the hub to convert. E.g. `gpt2` or `facebook/wav2vec2-base-960h`",
    )
    parser.add_argument(
        "-y",
        action="store_true",
        help="Ignore safety prompt",
    )
    parser.add_argument(
        "--src",
        type=str,
        default="PATH",
        help="path to torch checkpoint path",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="PATH",
        help="target file path to save the converted checkpoint",
    )
    args = parser.parse_args()

    # torch2ms
    convert_torch2ms(args.src, args.target)

    # bin2safetensors
    model_id = args.model_id
    api = HfApi()
    if args.y:
        txt = "y"
    commit_info, errors = convert_bin2safetensors(api, model_id, force=args.force)
    string = """### Success 🔥 Yay! This model was successfully converted"""
    if errors:
        string += "\nErrors during conversion:\n"
        string += "\n".join(f"Error while converting {filename}: {e}, skipped conversion" for filename, e in errors)
    print(string)
