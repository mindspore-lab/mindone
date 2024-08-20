import argparse
import os
from collections import defaultdict
from typing import Dict, List

import torch
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
                    f"Error while trying to find names to remove to save state dict, but found no suitable name\
                    to keep for saving amongst: {shared}. None is covering the entire storage.Refusing to save/load \
                    the model since you could be storing much more memory than needed. \
                    Please refer to https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an issue."
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


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )


def convert_file(
    pt_filename: str,
    sf_filename: str,
    discard_names: List[str],
):
    loaded = torch.load(pt_filename, map_location="cpu", weights_only=True)
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


def create_diff(pt_infos: Dict[str, List[str]], sf_infos: Dict[str, List[str]]) -> str:
    errors = []
    for key in ["missing_keys", "mismatched_keys", "unexpected_keys"]:
        pt_set = set(pt_infos[key])
        sf_set = set(sf_infos[key])

        pt_only = pt_set - sf_set
        sf_only = sf_set - pt_set

        if pt_only:
            errors.append(f"{key} : PT warnings contain {pt_only} which are not present in SF warnings")
        if sf_only:
            errors.append(f"{key} : SF warnings contain {sf_only} which are not present in PT warnings")
    return "\n".join(errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        "-s",
        type=str,
        help="path to source torch checkpoint, which ends with .pt",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        help="Filename to save. Specify folder, e.g., ./models, or file path which ends with .ckpt, e.g., ./models/dit.ckpt",
    )

    args = parser.parse_args()

    if not os.path.exists(args.src):
        raise ValueError(f"The provided source file {args.src} does not exist!")

    assert args.target.endswith(".safetensors"), "Must convert to .safetensors file"

    pt_path = args.src
    sf_path = args.target
    convert_file(pt_path, sf_path, discard_names=[])
