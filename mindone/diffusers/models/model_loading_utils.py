# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import importlib
import json
import os
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from huggingface_hub.utils import EntryNotFoundError

import mindspore as ms
from mindspore import nn

from ...safetensors.mindspore import load_file as safe_load_file
from ..utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_FILE_EXTENSION,
    WEIGHTS_INDEX_NAME,
    _add_variant,
    _get_model_file,
    logging,
)

logger = logging.get_logger(__name__)

_CLASS_REMAPPING_DICT = {
    "Transformer2DModel": {
        "ada_norm_zero": "DiTTransformer2DModel",
        "ada_norm_single": "PixArtTransformer2DModel",
    }
}


def _fetch_remapped_cls_from_config(config, old_class):
    previous_class_name = old_class.__name__
    remapped_class_name = _CLASS_REMAPPING_DICT.get(previous_class_name).get(config["norm_type"], None)

    # Details:
    # https://github.com/huggingface/diffusers/pull/7647#discussion_r1621344818
    if remapped_class_name:
        # load diffusers library to import compatible and original scheduler
        diffusers_library = importlib.import_module(__name__.split(".")[0] + ".diffusers")
        remapped_class = getattr(diffusers_library, remapped_class_name)
        logger.info(
            f"Changing class object to be of `{remapped_class_name}` type from `{previous_class_name}` type."
            f"This is because `{previous_class_name}` is scheduled to be deprecated in a future version. Note that this"
            " DOESN'T affect the final results."
        )
        return remapped_class
    else:
        return old_class


def load_state_dict(checkpoint_file: Union[str, os.PathLike], variant: Optional[str] = None):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        file_extension = os.path.basename(checkpoint_file).split(".")[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            return safe_load_file(checkpoint_file)
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


def _load_state_dict_into_model(model_to_load, state_dict: OrderedDict) -> List[str]:
    # TODO: error_msgs is always empty for now. Maybe we need to rewrite MindSpore's `load_param_into_net`.
    #  Error msgs should contain caught exception like size mismatch instead of missing/unexpected keys.
    # TODO: We should support loading float16 state_dict into float32 model, like PyTorch's behavior.
    error_msgs = []
    # TODO: State dict loading in mindspore does not cast dtype correctly. We do it manually. It's might unsafe.
    local_state = {k: v for k, v in model_to_load.parameters_and_names()}
    for k, v in state_dict.items():
        if k in local_state:
            v.set_dtype(local_state[k].dtype)
        else:
            pass  # unexpect key keeps origin dtype
    ms.load_param_into_net(model_to_load, state_dict, strict_load=True)
    return error_msgs


def _fetch_index_file(
    is_local,
    pretrained_model_name_or_path,
    subfolder,
    use_safetensors,
    cache_dir,
    variant,
    force_download,
    proxies,
    local_files_only,
    token,
    revision,
    user_agent,
    commit_hash,
):
    if is_local:
        index_file = Path(
            pretrained_model_name_or_path,
            subfolder or "",
            _add_variant(SAFE_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME, variant),
        )
    else:
        index_file_in_repo = Path(
            subfolder or "",
            _add_variant(SAFE_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME, variant),
        ).as_posix()
        try:
            index_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=index_file_in_repo,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=None,
                user_agent=user_agent,
                commit_hash=commit_hash,
            )
            index_file = Path(index_file)
        except (EntryNotFoundError, EnvironmentError):
            index_file = None

    return index_file


# ===============================================
# Sharded loading utils by huggingface Accelerate
# ===============================================

_SAFE_MODEL_NAME = "model"
_SAFE_WEIGHTS_NAME = f"{_SAFE_MODEL_NAME}.safetensors"


# Copied from mindone.transformers.modeling_utils.silence_mindspore_logger
@contextmanager
def silence_mindspore_logger():
    ms_logger = ms.log._get_logger()
    ms_level = ms_logger.level
    ms_logger.setLevel("ERROR")
    yield
    ms_logger.setLevel(ms_level)


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
        for checkpoint_file in checkpoint_files:
            loaded_checkpoint = load_state_dict(checkpoint_file)
            _ = _load_state_dict_into_model(model, loaded_checkpoint)
            unexpected_keys.update(set(loaded_checkpoint.keys()) - model_keys)
            del loaded_checkpoint
            gc.collect()

    if not strict and len(unexpected_keys) > 0:
        logger.warning(
            f"Some weights of the model checkpoint at {checkpoint} were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}. This may or may not be an issue - make sure that the checkpoint does not have unnecessary parameters, or that the model definition correctly corresponds to the checkpoint."  # noqa E501
        )

    return model


# =============================================
# Sharded saving by huggingface huggingface_hub
# =============================================

TensorT = TypeVar("TensorT")
TensorSizeFn_T = Callable[[TensorT], int]
StorageIDFn_T = Callable[[TensorT], Optional[Any]]

_MAX_SHARD_SIZE = "5GB"
_SAFETENSORS_WEIGHTS_FILE_PATTERN = "model{suffix}.safetensors"
_SIZE_UNITS = {
    "TB": 10**12,
    "GB": 10**9,
    "MB": 10**6,
    "KB": 10**3,
}


@dataclass
class StateDictSplit:
    is_sharded: bool = field(init=False)
    metadata: Dict[str, Any]
    filename_to_tensors: Dict[str, List[str]]
    tensor_to_filename: Dict[str, str]

    def __post_init__(self):
        self.is_sharded = len(self.filename_to_tensors) > 1


def split_state_dict_into_shards_factory(
    state_dict: Dict[str, TensorT],
    *,
    get_storage_size: TensorSizeFn_T,
    filename_pattern: str,
    get_storage_id: StorageIDFn_T = lambda tensor: None,
    max_shard_size: Union[int, str] = _MAX_SHARD_SIZE,
) -> StateDictSplit:
    """
    Split a model state dictionary in shards so that each shard is smaller than a given size.

    The shards are determined by iterating through the `state_dict` in the order of its keys. There is no optimization
    made to make each shard as close as possible to the maximum size passed. For example, if the limit is 10GB and we
    have tensors of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB], [6+2+2GB] and not
    [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's tensor is bigger than `max_shard_size`, it will end up in its own shard which will have a
    size greater than `max_shard_size`.

    </Tip>

    Args:
        state_dict (`Dict[str, Tensor]`):
            The state dictionary to save.
        get_storage_size (`Callable[[Tensor], int]`):
            A function that returns the size of a tensor when saved on disk in bytes.
        get_storage_id (`Callable[[Tensor], Optional[Any]]`, *optional*):
            A function that returns a unique identifier to a tensor storage. Multiple different tensors can share the
            same underlying storage. This identifier is guaranteed to be unique and constant for this tensor's storage
            during its lifetime. Two tensor storages with non-overlapping lifetimes may have the same id.
        filename_pattern (`str`, *optional*):
            The pattern to generate the files names in which the model will be saved. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
        max_shard_size (`int` or `str`, *optional*):
            The maximum size of each shard, in bytes. Defaults to 5GB.

    Returns:
        [`StateDictSplit`]: A `StateDictSplit` object containing the shards and the index to retrieve them.
    """
    storage_id_to_tensors: Dict[Any, List[str]] = {}

    shard_list: List[Dict[str, TensorT]] = []
    current_shard: Dict[str, TensorT] = {}
    current_shard_size = 0
    total_size = 0

    if isinstance(max_shard_size, str):
        max_shard_size = parse_size_to_int(max_shard_size)

    for key, tensor in state_dict.items():
        # when bnb serialization is used the weights in the state dict can be strings
        # check: https://github.com/huggingface/transformers/pull/24416 for more details
        if isinstance(tensor, str):
            logger.info("Skipping tensor %s as it is a string (bnb serialization)", key)
            continue

        # If a `tensor` shares the same underlying storage as another tensor, we put `tensor` in the same `block`
        storage_id = get_storage_id(tensor)
        if storage_id is not None:
            if storage_id in storage_id_to_tensors:
                # We skip this tensor for now and will reassign to correct shard later
                storage_id_to_tensors[storage_id].append(key)
                continue
            else:
                # This is the first tensor with this storage_id, we create a new entry
                # in the storage_id_to_tensors dict => we will assign the shard id later
                storage_id_to_tensors[storage_id] = [key]

        # Compute tensor size
        tensor_size = get_storage_size(tensor)

        # If this tensor is bigger than the maximal size, we put it in its own shard
        if tensor_size > max_shard_size:
            total_size += tensor_size
            shard_list.append({key: tensor})
            continue

        # If this tensor is going to tip up over the maximal size, we split.
        # Current shard already has some tensors, we add it to the list of shards and create a new one.
        if current_shard_size + tensor_size > max_shard_size:
            shard_list.append(current_shard)
            current_shard = {}
            current_shard_size = 0

        # Add the tensor to the current shard
        current_shard[key] = tensor
        current_shard_size += tensor_size
        total_size += tensor_size

    # Add the last shard
    if len(current_shard) > 0:
        shard_list.append(current_shard)
    nb_shards = len(shard_list)

    # Loop over the tensors that share the same storage and assign them together
    for storage_id, keys in storage_id_to_tensors.items():
        # Let's try to find the shard where the first tensor of this storage is and put all tensors in the same shard
        for shard in shard_list:
            if keys[0] in shard:
                for key in keys:
                    shard[key] = state_dict[key]
                break

    # If we only have one shard, we return it => no need to build the index
    if nb_shards == 1:
        filename = filename_pattern.format(suffix="")
        return StateDictSplit(
            metadata={"total_size": total_size},
            filename_to_tensors={filename: list(state_dict.keys())},
            tensor_to_filename={key: filename for key in state_dict.keys()},
        )

    # Now that each tensor is assigned to a shard, let's assign a filename to each shard
    tensor_name_to_filename = {}
    filename_to_tensors = {}
    for idx, shard in enumerate(shard_list):
        filename = filename_pattern.format(suffix=f"-{idx+1:05d}-of-{nb_shards:05d}")
        for key in shard:
            tensor_name_to_filename[key] = filename
        filename_to_tensors[filename] = list(shard.keys())

    # Build the index and return
    return StateDictSplit(
        metadata={"total_size": total_size},
        filename_to_tensors=filename_to_tensors,
        tensor_to_filename=tensor_name_to_filename,
    )


def parse_size_to_int(size_as_str: str) -> int:
    """
    Parse a size expressed as a string with digits and unit (like `"5MB"`) to an integer (in bytes).

    Supported units are "TB", "GB", "MB", "KB".

    Args:
        size_as_str (`str`): The size to convert. Will be directly returned if an `int`.

    Example:

    ```py
    >>> parse_size_to_int("5MB")
    5000000
    ```
    """
    size_as_str = size_as_str.strip()

    # Parse unit
    unit = size_as_str[-2:].upper()
    if unit not in _SIZE_UNITS:
        raise ValueError(f"Unit '{unit}' not supported. Supported units are TB, GB, MB, KB. Got '{size_as_str}'.")
    multiplier = _SIZE_UNITS[unit]

    # Parse value
    try:
        value = float(size_as_str[:-2].strip())
    except ValueError as e:
        raise ValueError(f"Could not parse the size value from '{size_as_str}': {e}") from e

    return int(value * multiplier)


def split_torch_state_dict_into_shards(
    state_dict: Dict[str, "ms.Tensor"],
    *,
    filename_pattern: str = _SAFETENSORS_WEIGHTS_FILE_PATTERN,
    max_shard_size: Union[int, str] = _MAX_SHARD_SIZE,
) -> StateDictSplit:
    """
    Split a model state dictionary in shards so that each shard is smaller than a given size.

    The shards are determined by iterating through the `state_dict` in the order of its keys. There is no optimization
    made to make each shard as close as possible to the maximum size passed. For example, if the limit is 10GB and we
    have tensors of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB], [6+2+2GB] and not
    [6+2+2GB], [6+2GB], [6GB].


    <Tip>

    To save a model state dictionary to the disk, see [`save_torch_state_dict`]. This helper uses
    `split_torch_state_dict_into_shards` under the hood.

    </Tip>

    <Tip warning={true}>

    If one of the model's tensor is bigger than `max_shard_size`, it will end up in its own shard which will have a
    size greater than `max_shard_size`.

    </Tip>

    Args:
        state_dict (`Dict[str, ms.Tensor]`):
            The state dictionary to save.
        filename_pattern (`str`, *optional*):
            The pattern to generate the files names in which the model will be saved. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
            Defaults to `"model{suffix}.safetensors"`.
        max_shard_size (`int` or `str`, *optional*):
            The maximum size of each shard, in bytes. Defaults to 5GB.

    Returns:
        [`StateDictSplit`]: A `StateDictSplit` object containing the shards and the index to retrieve them.

    Example:
    ```py
    >>> import json
    >>> import os
    >>> from safetensors.torch import save_file as safe_save_file
    >>> from huggingface_hub import split_torch_state_dict_into_shards

    >>> def save_state_dict(state_dict: Dict[str, ms.Tensor], save_directory: str):
    ...     state_dict_split = split_torch_state_dict_into_shards(state_dict)
    ...     for filename, tensors in state_dict_split.filename_to_tensors.items():
    ...         shard = {tensor: state_dict[tensor] for tensor in tensors}
    ...         safe_save_file(
    ...             shard,
    ...             os.path.join(save_directory, filename),
    ...             metadata={"format": "pt"},
    ...         )
    ...     if state_dict_split.is_sharded:
    ...         index = {
    ...             "metadata": state_dict_split.metadata,
    ...             "weight_map": state_dict_split.tensor_to_filename,
    ...         }
    ...         with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
    ...             f.write(json.dumps(index, indent=2))
    ```
    """
    return split_state_dict_into_shards_factory(
        state_dict,
        max_shard_size=max_shard_size,
        filename_pattern=filename_pattern,
        get_storage_size=get_torch_storage_size,
    )


def get_torch_storage_size(tensor: "ms.Tensor") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/08db34094e9e59e2f9218f2df133b7b4aaff5a99/bindings/python/py_src/safetensors/torch.py#L31C1-L41C59
    """
    return tensor.nelement() * _get_dtype_size(tensor.dtype)


@lru_cache()
def _get_dtype_size(dtype: "ms.Type") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/08db34094e9e59e2f9218f2df133b7b4aaff5a99/bindings/python/py_src/safetensors/torch.py#L344
    """
    import mindspore as ms

    _SIZE = {
        ms.int64: 8,
        ms.float32: 4,
        ms.int32: 4,
        ms.bfloat16: 2,
        ms.float16: 2,
        ms.int16: 2,
        ms.uint8: 1,
        ms.int8: 1,
        ms.bool_: 1,
        ms.float64: 8,
    }
    return _SIZE[dtype]
