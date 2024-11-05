"""
Modified from
https://github.com/huggingface/safetensors/blob/main/bindings/python/convert.py
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set

import requests
import torch
from huggingface_hub import HfApi, configure_http_backend, hf_hub_download, snapshot_download
from safetensors.torch import _find_shared_tensors, _is_complete, load_file, save_file


def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session


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
                    "Error while trying to find names to remove to save state dict, but found no suitable name to keep"
                    f" for saving amongst: {shared}. None is covering the entire storage.Refusing to save/load the model"
                    " since you could be storing much more memory than needed."
                    " Please refer to https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an issue."
                )

        keep_name = sorted(list(complete_names))[0]

        # Mechanism to preferentially select keys to keep
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


def get_discard_names(
    model_id: str, revision: Optional[str], folder: str, token: Optional[str], endpoint: str
) -> List[str]:
    try:
        import json

        import transformers

        config_filename = hf_hub_download(
            model_id, revision=revision, filename="config.json", token=token, cache_dir=folder, endpoint=endpoint
        )
        with open(config_filename, "r") as f:
            config = json.load(f)
        architecture = config["architectures"][0]

        class_ = getattr(transformers, architecture)

        # Name for this variable depends on transformers version.
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
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )


def rename(pt_filename: str) -> str:
    filename, ext = os.path.splitext(pt_filename)
    local = f"{filename}.safetensors"
    local = local.replace("pytorch_model", "model")
    return local


def convert_multi(
    model_id: str, *, revision=Optional[str], folder: str, token: Optional[str], discard_names: List[str], endpoint: str
) -> str:
    filename = hf_hub_download(
        repo_id=model_id,
        revision=revision,
        filename="pytorch_model.bin.index.json",
        token=token,
        cache_dir=folder,
        endpoint=endpoint,
    )
    save_path = os.path.dirname(filename)
    with open(filename, "r") as f:
        data = json.load(f)

    filenames = set(data["weight_map"].values())
    for filename in filenames:
        pt_filename = hf_hub_download(
            model_id, revision=revision, filename=filename, token=token, cache_dir=folder, endpoint=endpoint
        )
        sf_filename = rename(pt_filename)
        sf_filename = os.path.join(save_path, sf_filename)
        convert_file(pt_filename, sf_filename, discard_names=discard_names)

    index = os.path.join(save_path, "model.safetensors.index.json")
    with open(index, "w") as f:
        newdata = {k: v for k, v in data.items()}
        newmap = {k: rename(v) for k, v in data["weight_map"].items()}
        newdata["weight_map"] = newmap
        json.dump(newdata, f, indent=4)

    return save_path


def convert_single(
    model_id: str,
    *,
    revision: Optional[str],
    folder: str,
    token: Optional[str],
    discard_names: List[str],
    endpoint: str,
) -> str:
    pt_filename = hf_hub_download(
        repo_id=model_id,
        revision=revision,
        filename="pytorch_model.bin",
        token=token,
        cache_dir=folder,
        endpoint=endpoint,
    )
    save_path = os.path.dirname(pt_filename)
    sf_name = "model.safetensors"
    sf_filename = os.path.join(save_path, sf_name)
    convert_file(pt_filename, sf_filename, discard_names)
    return save_path


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


def convert_generic(
    model_id: str, *, revision=Optional[str], folder: str, filenames: Set[str], token: Optional[str], endpoint: str
) -> str:
    save_path = ""
    extensions = {".bin", ".ckpt"}
    for filename in filenames:
        prefix, ext = os.path.splitext(filename)
        if ext in extensions:
            pt_filename = hf_hub_download(
                model_id, revision=revision, filename=filename, token=token, cache_dir=folder, endpoint=endpoint
            )
            save_path = os.path.dirname(pt_filename)

            dirname, raw_filename = os.path.split(filename)
            if raw_filename == "pytorch_model.bin":
                # XXX: This is a special case to handle `transformers` and the
                # `transformers` part of the model which is actually loaded by `transformers`.
                sf_in_repo = os.path.join(dirname, "model.safetensors")
            else:
                sf_in_repo = f"{prefix}.safetensors"
            sf_filename = os.path.join(save_path, sf_in_repo)
            convert_file(pt_filename, sf_filename, discard_names=[])
    return save_path


def convert(
    model_id: str,
    revision: Optional[str] = None,
    folder: str = None,
    force: bool = False,
    endpoint: str = "https://hf-mirror.com",
) -> str:
    api = HfApi(endpoint=endpoint)
    info = api.model_info(model_id, revision=revision)
    filenames = set(s.rfilename for s in info.siblings)

    library_name = getattr(info, "library_name", None)
    if any(filename.endswith(".safetensors") for filename in filenames) and not force:
        print(f"Model {model_id} is already converted. Downloading safetensors...")
        save_path = snapshot_download(  # Download an entire directory, including the tokenizer config
            model_id,
            revision=revision,
            allow_patterns=["*.safetensors", "*.json", "*.model"],
            token=api.token,
            cache_dir=folder,
            endpoint=endpoint,
        )
    else:
        snapshot_download(  # Download an entire directory, including the tokenizer config
            model_id,
            revision=revision,
            allow_patterns=["*.bin", "*.json", "*.model"],
            token=api.token,
            cache_dir=folder,
            endpoint=endpoint,
        )
        if library_name == "transformers":
            discard_names = get_discard_names(
                model_id, revision=revision, folder=folder, token=api.token, endpoint=endpoint
            )
            if "pytorch_model.bin" in filenames:
                save_path = convert_single(
                    model_id,
                    revision=revision,
                    folder=folder,
                    token=api.token,
                    discard_names=discard_names,
                    endpoint=endpoint,
                )
            elif "pytorch_model.bin.index.json" in filenames:
                save_path = convert_multi(
                    model_id,
                    revision=revision,
                    folder=folder,
                    token=api.token,
                    discard_names=discard_names,
                    endpoint=endpoint,
                )
            else:
                raise RuntimeError(f"Model {model_id} doesn't seem to be a valid pytorch model. Cannot convert")
        else:
            save_path = convert_generic(
                model_id, revision=revision, folder=folder, filenames=filenames, token=api.token, endpoint=endpoint
            )
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloads and converts weights to `safetensors` format.")
    parser.add_argument(
        "model_id",
        type=str,
        help="The name of the model on the hub to convert. E.g. `gpt2` or `facebook/wav2vec2-base-960h`",
    )
    parser.add_argument(
        "--revision",
        type=str,
        help="The revision to convert",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory to download and save the converted model",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="The Huggingface endpoint to use. Defaults to `https://hf-mirror.com`.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force weights re-conversion.",
    )
    parser.add_argument(
        "--disable_ssl_verify",
        action="store_true",
        help="Disable SSL verification when downloading the model weights.",
    )

    args = parser.parse_args()
    if args.disable_ssl_verify:
        configure_http_backend(backend_factory=backend_factory)

    path = convert(
        args.model_id, revision=args.revision, folder=args.output_dir, force=args.force, endpoint=args.endpoint
    )
    print(f"Converted weights saved to {os.path.dirname(os.path.dirname(path))}")
