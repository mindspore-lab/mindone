import gc
import json
import os
from typing import Dict, Optional, Union

import safetensors
from tqdm import tqdm

import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter, Tensor

from mindone.diffusers.models.model_loading_utils import (
    _SAFE_WEIGHTS_NAME,
    SAFETENSORS_FILE_EXTENSION,
    logger,
    nullcontext,
    silence_mindspore_logger,
)


def safe_load_file(filename: Union[str, os.PathLike], dtype: ms.Type = ms.float32) -> Dict[str, ms.Tensor]:
    output = dict()
    with safetensors.safe_open(filename, framework="np") as f:
        for k in f.keys():
            output[k] = Parameter(Tensor(f.get_tensor(k), dtype=dtype), name=k)
    return output


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike], variant: Optional[str] = None, dtype: ms.Type = ms.float32
):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        file_extension = os.path.basename(checkpoint_file).split(".")[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            return safe_load_file(checkpoint_file, dtype=dtype)
        else:
            raise NotImplementedError(
                f"Only supports deserialization of weights file in safetensors format, but got {checkpoint_file}"
            )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' " f"at '{checkpoint_file}'. "
            )


def load_checkpoint_and_dispatch(
    model: nn.Cell,
    checkpoint: Union[str, os.PathLike],
    dtype: Optional[Union[str, ms.Type]] = None,
    strict: bool = False,
):
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded and adds the various hooks that will make this model run properly (even if split across devices).

    Args:
        model (`mindspore.nn.Cell`): The model in which we want to load a checkpoint.
        checkpoint (`str` or `os.PathLike`):
            The folder checkpoint to load. It can be:
            - a path to a file containing a whole model state dict
            - a path to a `.json` file containing the index to a sharded checkpoint
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        dtype (`str` or `mindspore.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        force_hooks (`bool`, *optional*, defaults to `False`):
            Whether or not to force device hooks to be attached to the model even if all layers are dispatched to a
            single device.
        strict (`bool`, *optional*, defaults to `False`):
            Whether to strictly enforce that the keys in the checkpoint state_dict match the keys of the model's
            state_dict.

    Example:

    ```python
    >>> from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    >>> from huggingface_hub import hf_hub_download
    >>> from transformers import AutoConfig, AutoModelForCausalLM

    >>> # Download the Weights
    >>> checkpoint = "EleutherAI/gpt-j-6B"
    >>> weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")

    >>> # Create a model and initialize it with empty weights
    >>> config = AutoConfig.from_pretrained(checkpoint)
    >>> with init_empty_weights():
    ...     model = AutoModelForCausalLM.from_config(config)

    >>> # Load the checkpoint and dispatch it to the right devices
    >>> model = load_checkpoint_and_dispatch(
    ...     model, weights_location, device_map="auto", no_split_module_classes=["GPTJBlock"]
    ... )
    ```
    """

    if isinstance(dtype, str):
        # We accept "torch.float16" or just "float16"
        dtype = dtype.replace("mindspore.", "")
        dtype = getattr(ms, dtype)

    checkpoint_files = None
    index_filename = None
    if os.path.isfile(checkpoint):
        if str(checkpoint).endswith(".json"):
            index_filename = checkpoint
        else:
            checkpoint_files = [checkpoint]
    elif os.path.isdir(checkpoint):
        # check if the whole state dict is present
        potential_state_safetensor = [f for f in os.listdir(checkpoint) if f == _SAFE_WEIGHTS_NAME]
        if len(potential_state_safetensor) == 1:
            checkpoint_files = [os.path.join(checkpoint, potential_state_safetensor[0])]
        else:
            # otherwise check for sharded checkpoints
            potential_index = [f for f in os.listdir(checkpoint) if f.endswith(".index.json")]
            if len(potential_index) == 0:
                raise ValueError(
                    f"{checkpoint} is not a folder containing a `.index.json` file or a {_SAFE_WEIGHTS_NAME} file"
                )
            elif len(potential_index) == 1:
                index_filename = os.path.join(checkpoint, potential_index[0])
            else:
                raise ValueError(
                    f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones."
                )
    else:
        raise ValueError(
            "`checkpoint` should be the path to a file containing a whole state dict, or the index of a sharded "
            f"checkpoint, or a folder containing a sharded checkpoint or the whole state dict, but got {checkpoint}."
        )

    if index_filename is not None:
        checkpoint_folder = os.path.split(index_filename)[0]
        with open(index_filename) as f:
            index = json.loads(f.read())

        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]

    # Logic for missing/unexepected keys goes here.
    unexpected_keys = set()
    model_keys = set(model.parameters_dict().keys())
    is_sharded = index_filename is not None
    cm = silence_mindspore_logger() if is_sharded else nullcontext()
    with cm:
        for checkpoint_file in tqdm(checkpoint_files, desc="Loading safetensors"):
            loaded_checkpoint = load_state_dict(checkpoint_file, dtype=dtype)
            ms.load_param_into_net(model, loaded_checkpoint, strict_load=True)
            unexpected_keys.update(set(loaded_checkpoint.keys()) - model_keys)
            del loaded_checkpoint
            gc.collect()

    if not strict and len(unexpected_keys) > 0:
        logger.warning(
            f"Some weights of the model checkpoint at {checkpoint} were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}. This may or may not be an issue - make sure that the checkpoint does not have unnecessary parameters, or that the model definition correctly corresponds to the checkpoint."  # noqa E501
        )

    return model
