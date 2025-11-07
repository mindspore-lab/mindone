# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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

import copy
import functools
import inspect
import json
import os
import re
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple, Type, Union

from huggingface_hub import DDUFEntry, create_repo
from huggingface_hub.utils import validate_hf_hub_args
from typing_extensions import Self

import mindspore as ms
from mindspore import mint, nn
from mindspore.nn.utils import no_init_parameters

from mindone.safetensors.mindspore import save_file as safe_save_file
from mindone.utils.modeling_patch import patch_nn_default_dtype, unpatch_nn_default_dtype

from .. import __version__
from ..utils import (
    CONFIG_NAME,
    HF_ENABLE_PARALLEL_LOADING,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    _add_variant,
    _get_checkpoint_shard_files,
    _get_model_file,
    deprecate,
    logging,
)
from ..utils.hub_utils import PushToHubMixin, load_or_create_model_card, populate_model_card
from .model_loading_utils import (
    _fetch_index_file,
    _fetch_index_file_legacy,
    _load_shard_file,
    _load_shard_files_with_threadpool,
    load_state_dict,
    split_torch_state_dict_into_shards,
)

ms.Parameter._data = ms.Tensor.data
ms.Parameter.data_ptr = ms.Tensor.data_ptr


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


logger = logging.get_logger(__name__)

_REGEX_SHARD = re.compile(r"(.*?)-\d{5}-of-\d{5}")


def _get_pt2ms_mappings(m):
    mappings = {}  # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)
    for name, cell in m.cells_and_names():
        if isinstance(cell, (nn.Conv1d, nn.Conv1dTranspose)):
            mappings[f"{name}.weight"] = f"{name}.weight", lambda x: ms.Parameter(x.unsqueeze(dim=-2), name=x.name)
            if "weight_norm_cell" in name:
                ori_name = name.replace(".weight_norm_cell", "")
                mappings[f"{ori_name}.weight_g"] = f"{ori_name}.weight_g", lambda x: ms.Parameter(
                    x.unsqueeze(dim=-2), name=x.name
                )
                mappings[f"{ori_name}.weight_v"] = f"{ori_name}.weight_v", lambda x: ms.Parameter(
                    x.unsqueeze(dim=-2), name=x.name
                )
                mappings[f"{ori_name}.bias"] = f"{name}.bias", lambda x: x
        elif isinstance(cell, nn.Embedding):
            mappings[f"{name}.weight"] = f"{name}.embedding_table", lambda x: x
        elif isinstance(cell, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            mappings[f"{name}.weight"] = f"{name}.gamma", lambda x: x
            mappings[f"{name}.bias"] = f"{name}.beta", lambda x: x
            if isinstance(
                cell,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                ),
            ):
                mappings[f"{name}.running_mean"] = f"{name}.moving_mean", lambda x: x
                mappings[f"{name}.running_var"] = f"{name}.moving_variance", lambda x: x
                mappings[f"{name}.num_batches_tracked"] = None, lambda x: x
        elif isinstance(cell, mint.nn.BatchNorm2d):
            mappings[f"{name}.num_batches_tracked"] = None, lambda x: x.to(ms.float32)
    return mappings


def _convert_state_dict(m, state_dict_pt):
    if not state_dict_pt:
        return state_dict_pt
    pt2ms_mappings = _get_pt2ms_mappings(m)
    state_dict_ms = {}
    while state_dict_pt:
        name_pt, data_pt = state_dict_pt.popitem()
        name_ms, data_mapping = pt2ms_mappings.get(name_pt, (name_pt, lambda x: x))
        data_ms = data_mapping(data_pt)
        if name_ms is not None:
            state_dict_ms[name_ms] = data_ms
    return state_dict_ms


def get_parameter_dtype(parameter: nn.Cell) -> ms.Type:
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None

    for name, param in parameter.parameters_and_names():
        last_dtype = param.dtype
        if (
            hasattr(parameter, "_keep_in_fp32_modules")
            and parameter._keep_in_fp32_modules
            and any(m in name for m in parameter._keep_in_fp32_modules)
        ):
            continue

        if param.is_floating_point():
            return param.dtype

    for buffer in parameter.buffers():
        last_dtype = buffer.dtype
        if buffer.is_floating_point():
            return buffer.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype


class ModelMixin(nn.Cell, PushToHubMixin):
    r"""
    Base class for all models.

    [`ModelMixin`] takes care of storing the model configuration and provides methods for loading, downloading and
    saving models.

        - **config_name** ([`str`]) -- Filename to save a model to when calling [`~models.ModelMixin.save_pretrained`].
    """

    config_name = CONFIG_NAME
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    _supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = None
    _no_split_modules = None
    _keep_in_fp32_modules = None
    _skip_layerwise_casting_patterns = None
    _supports_group_offloading = True
    _repeated_blocks = []

    def __init__(self):
        super().__init__()

        self._gradient_checkpointing_func = None

    def __getattr__(self, name: str) -> Any:
        """The only reason we overwrite `getattr` here is to gracefully deprecate accessing
        config attributes directly. See https://github.com/huggingface/diffusers/pull/3129 We need to overwrite
        __getattr__ here in addition so that we don't trigger `nn.Cell`'s __getattr__':
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        """

        is_in_config = "_internal_dict" in self.__dict__ and hasattr(self.__dict__["_internal_dict"], name)
        is_attribute = name in self.__dict__

        if is_in_config and not is_attribute:
            deprecation_message = f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'unet.config.{name}'."  # noqa: E501
            deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False, stacklevel=3)
            return self._internal_dict[name]

        # call PyTorch's https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        return super().__getattr__(name)

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.
        """
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for _, m in self.cells_and_names())

    def enable_gradient_checkpointing(self, gradient_checkpointing_func: Optional[Callable] = None) -> None:
        """
        Activates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).

        Args:
            gradient_checkpointing_func (`Callable`, *optional*):
                The function to use for gradient checkpointing. If `None`, the default MindSpore checkpointing function
                is used (`mindspore.nn.Cell.recompute_`).
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(
                f"{self.__class__.__name__} does not support gradient checkpointing. Please make sure to set the boolean attribute "
                f"`_supports_gradient_checkpointing` to `True` in the class definition."
            )

        if gradient_checkpointing_func is None:

            def _gradient_checkpointing_func(module, *args):
                module.recompute_(mode=True)
                return module

            gradient_checkpointing_func = _gradient_checkpointing_func

        self._set_gradient_checkpointing(enable=True)

    def disable_gradient_checkpointing(self) -> None:
        """
        Deactivates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).
        """
        if self._supports_gradient_checkpointing:
            self._set_gradient_checkpointing(enable=False)

    def enable_flash_sdp(self, enabled: bool):
        r"""
        .. warning:: This flag is beta and subject to change.

        Enables or disables flash scaled dot product attention.
        """

        # Recursively walk through all the children.
        # Any children which exposes the enable_flash_sdp method
        # gets the message
        def fn_recursive_set_mem_eff(module: nn.Cell):
            if hasattr(module, "enable_flash_sdp"):
                module.enable_flash_sdp(enabled)

            for child in module.cells():
                fn_recursive_set_mem_eff(child)

        for module in self.cells():
            if isinstance(module, nn.Cell):
                fn_recursive_set_mem_eff(module)

    def set_flash_attention_force_cast_dtype(self, force_cast_dtype: Optional[ms.Type]):
        r"""
        Since the flash-attention operator in MindSpore only supports float16 and bfloat16 data types, we need to manually
        set whether to force data type conversion.

        When the attention interface encounters data of an unsupported data type,
        if `force_cast_dtype` is not None, the function will forcibly convert the data to `force_cast_dtype` for computation
        and then restore it to the original data type afterward. If `force_cast_dtype` is None, it will fall back to the
        original attention calculation using mathematical formulas.

        Parameters:
            force_cast_dtype (Optional): The data type to which the input data should be forcibly converted. If None, no forced
            conversion is performed.
        """

        # Recursively walk through all the children.
        # Any children which exposes the set_flash_attention_force_cast_dtype method
        # gets the message
        def fn_recursive_set_mem_eff(module: nn.Cell):
            if hasattr(module, "set_flash_attention_force_cast_dtype"):
                module.set_flash_attention_force_cast_dtype(force_cast_dtype)

            for child in module.cells():
                fn_recursive_set_mem_eff(child)

        for module in self.cells():
            if isinstance(module, nn.Cell):
                fn_recursive_set_mem_eff(module)

    def set_use_memory_efficient_attention_xformers(self, valid: bool, attention_op: Optional[Callable] = None) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: nn.Cell):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.cells():
                fn_recursive_set_mem_eff(child)

        for module in self.cells():
            if isinstance(module, nn.Cell):
                fn_recursive_set_mem_eff(module)

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None) -> None:
        r"""
        Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).

        When this option is enabled, you should observe lower GPU memory usage and a potential speed up during
        inference. Speed up during training is not guaranteed.

        <Tip warning={true}>

        ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
        precedent.

        </Tip>

        Parameters:
            attention_op (`Callable`, *optional*):
                Not supported for now.

        Examples:

        ```py
        >>> import mindspore as ms
        >>> from mindone.diffusers import UNet2DConditionModel

        >>> model = UNet2DConditionModel.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1", subfolder="unet", mindspore_dtype=ms.float16
        ... )
        >>> model.enable_xformers_memory_efficient_attention()
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self) -> None:
        r"""
        Disable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).
        """
        self.set_use_memory_efficient_attention_xformers(False)

    def enable_layerwise_casting(
        self,
        storage_dtype: ms.Type,
        compute_dtype: Optional[ms.Type] = None,
        skip_modules_pattern: Optional[Tuple[str, ...]] = None,
        skip_modules_classes: Optional[Tuple[Type[nn.Cell], ...]] = None,
        non_blocking: bool = False,
    ) -> None:
        r"""
        Activates layerwise casting for the current model.

        Layerwise casting is a technique that casts the model weights to a lower precision dtype for storage but
        upcasts them on-the-fly to a higher precision dtype for computation. This process can significantly reduce the
        memory footprint from model weights, but may lead to some quality degradation in the outputs. Most degradations
        are negligible, mostly stemming from weight casting in normalization and modulation layers.

        By default, most models in diffusers set the `_skip_layerwise_casting_patterns` attribute to ignore patch
        embedding, positional embedding and normalization layers. This is because these layers are most likely
        precision-critical for quality. If you wish to change this behavior, you can set the
        `_skip_layerwise_casting_patterns` attribute to `None`, or call
        [`~hooks.layerwise_casting.apply_layerwise_casting`] with custom arguments.

        Example:
            Using [`~models.ModelMixin.enable_layerwise_casting`]:

            ```python
            >>> from mindone.diffusers import CogVideoXTransformer3DModel

            >>> transformer = CogVideoXTransformer3DModel.from_pretrained(
            ...     "THUDM/CogVideoX-5b", subfolder="transformer", mindspore_dtype=ms.bfloat16
            ... )

            >>> # Enable layerwise casting via the model, which ignores certain modules by default
            >>> transformer.enable_layerwise_casting(storage_dtype=ms.float8_e4m3fn, compute_dtype=ms.bfloat16)
            ```

        Args:
            storage_dtype (`mindspore.Type`):
                The dtype to which the model should be cast for storage.
            compute_dtype (`mindspore.Type`):
                The dtype to which the model weights should be cast during the forward pass.
            skip_modules_pattern (`Tuple[str, ...]`, *optional*):
                A list of patterns to match the names of the modules to skip during the layerwise casting process. If
                set to `None`, default skip patterns are used to ignore certain internal layers of modules and PEFT
                layers.
            skip_modules_classes (`Tuple[Type[nn.Cell], ...]`, *optional*):
                A list of module classes to skip during the layerwise casting process.
            non_blocking (`bool`, *optional*, defaults to `False`):
                If `True`, the weight casting operations are non-blocking.
        """
        raise NotImplementedError("`enable_layerwise_casting` is not yet supported.")

    def enable_group_offload(
        self,
        onload_device: str = "Ascend",
        offload_device: str = "CPU",
        offload_type: str = "block_level",
        num_blocks_per_group: Optional[int] = None,
        non_blocking: bool = False,
        use_stream: bool = False,
        record_stream: bool = False,
        low_cpu_mem_usage=False,
    ) -> None:
        r"""
        Activates group offloading for the current model.

        See [`~hooks.group_offloading.apply_group_offloading`] for more information.

        Example:

            ```python
            >>> from mindone.diffusers import CogVideoXTransformer3DModel

            >>> transformer = CogVideoXTransformer3DModel.from_pretrained(
            ...     "THUDM/CogVideoX-5b", subfolder="transformer", mindspore_dtype=ms.bfloat16
            ... )

            >>> transformer.enable_group_offload(
            ...     onload_device="Ascend",
            ...     offload_device="CPU",
            ...     offload_type="leaf_level",
            ...     use_stream=True,
            ... )
            ```
        """
        raise NotImplementedError("`enable_group_offload` is not yet supported.")

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "10GB",
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory so that it can be reloaded using the
        [`~models.ModelMixin.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a model and its configuration file to. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `mindspore.save_checkpoint` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
            max_shard_size (`int` or `str`, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5GB"`).
                If expressed as an integer, the unit is bytes. Note that this limit will be decreased after a certain
                period of time (starting from Oct 2024) to allow users to upgrade to the latest version of `diffusers`.
                This is to establish a common default size for this argument across different libraries in the Hugging
                Face ecosystem (`transformers`, and `accelerate`, for example).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        weights_name = SAFETENSORS_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        weights_name = _add_variant(weights_name, variant)
        weights_name_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(
            ".safetensors", "{suffix}.safetensors"
        )

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

        # Only save the model itself if we are using distributed training
        model_to_save = self

        # Attach architecture to the config
        # Save the config
        if is_main_process:
            model_to_save.save_config(save_directory)

        # Save the model
        state_dict = {k: v for k, v in model_to_save.parameters_and_names()}

        # Save the model
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, max_shard_size=max_shard_size, filename_pattern=weights_name_pattern
        )

        # Clean the folder from a previous save
        if is_main_process:
            for filename in os.listdir(save_directory):
                if filename in state_dict_split.filename_to_tensors.keys():
                    continue
                full_filename = os.path.join(save_directory, filename)
                if not os.path.isfile(full_filename):
                    continue
                weights_without_ext = weights_name_pattern.replace(".bin", "").replace(".safetensors", "")
                weights_without_ext = weights_without_ext.replace("{suffix}", "")
                filename_without_ext = filename.replace(".bin", "").replace(".safetensors", "")
                # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
                if (
                    filename.startswith(weights_without_ext)
                    and _REGEX_SHARD.fullmatch(filename_without_ext) is not None
                ):
                    os.remove(full_filename)

        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
            filepath = os.path.join(save_directory, filename)
            if safe_serialization:
                # At some point we will need to deal better with save_function (used for TPU and other distributed
                # joyfulness), but for now this enough.
                safe_save_file(shard, filepath, metadata={"format": "np"})
            else:
                ms.save_checkpoint(shard, filepath)

        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )
        else:
            path_to_weights = os.path.join(save_directory, weights_name)
            logger.info(f"Model weights saved in {path_to_weights}")

        if push_to_hub:
            # Create a new empty model card and eventually tag it
            model_card = load_or_create_model_card(repo_id, token=token)
            model_card = populate_model_card(model_card)
            model_card.save(Path(save_directory, "README.md").as_posix())

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs) -> Self:
        r"""
        Instantiate a pretrained PyTorch model from a pretrained model configuration.

        The model is set in evaluation mode - `model.eval()` - by default, and dropout modules are deactivated. To
        train the model, set it back in training mode with `model.train()`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`~ModelMixin.save_pretrained`].

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            mindspore_dtype (`mindspore.Type`, *optional*):
                Override the default `mindspore.Type` and load the model with another dtype.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info (`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            variant (`str`, *optional*):
                Load weights from a specified `variant` filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the `safetensors` weights are downloaded if they're available **and** if the
                `safetensors` library is installed. If set to `True`, the model is forcibly loaded from `safetensors`
                weights. If set to `False`, `safetensors` weights are not loaded.
            disable_mmap ('bool', *optional*, defaults to 'False'):
                Whether to disable mmap when loading a Safetensors model. This option can perform better when the model
                is on a network mount or hard drive, which may not handle the seeky-ness of mmap very well.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with `hf
        auth login`. You can also activate the special
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a
        firewalled environment.

        </Tip>

        Example:

        ```py
        from mindone.diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        ```

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```bash
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at
        runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        mindspore_dtype = kwargs.pop("mindspore_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        dduf_entries: Optional[Dict[str, DDUFEntry]] = kwargs.pop("dduf_entries", None)
        disable_mmap = kwargs.pop("disable_mmap", False)

        is_parallel_loading_enabled = HF_ENABLE_PARALLEL_LOADING
        if is_parallel_loading_enabled:
            raise NotImplementedError("Parallel loading is not supported.")

        if mindspore_dtype is not None and not isinstance(mindspore_dtype, ms.Type):
            mindspore_dtype = ms.float32
            logger.warning(
                f"Passed `mindspore_dtype` {mindspore_dtype} is not a `ms.Type`. Defaulting to `ms.float32`."
            )

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }
        unused_kwargs = {}

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            dduf_entries=dduf_entries,
            **kwargs,
        )
        # no in-place modification of the original config.
        config = copy.deepcopy(config)

        # Check if `_keep_in_fp32_modules` is not None
        # use_keep_in_fp32_modules = cls._keep_in_fp32_modules is not None and (
        #     hf_quantizer is None or getattr(hf_quantizer, "use_keep_in_fp32_modules", False)
        # )

        # if use_keep_in_fp32_modules:
        #     keep_in_fp32_modules = cls._keep_in_fp32_modules
        #     if not isinstance(keep_in_fp32_modules, list):
        #         keep_in_fp32_modules = [keep_in_fp32_modules]

        #     if low_cpu_mem_usage is None:
        #         low_cpu_mem_usage = True
        #         logger.info("Set `low_cpu_mem_usage` to True as `_keep_in_fp32_modules` is not None.")
        #     elif not low_cpu_mem_usage:
        #         raise ValueError("`low_cpu_mem_usage` cannot be False when `keep_in_fp32_modules` is True.")
        # else:
        #     keep_in_fp32_modules = []

        # Check if `_keep_in_fp32_modules` is not None
        # use_keep_in_fp32_modules = cls._keep_in_fp32_modules is not None

        # FIXME: In MindONE we don't support `low_cpu_mem_usage`,
        # which is required to selectively keep some modules in fp32.
        # Therefore, we disable this feature by setting `keep_in_fp32_modules = []`.
        keep_in_fp32_modules = []

        is_sharded = False
        resolved_model_file = None

        # Determine if we're loading from a directory of sharded checkpoints.
        sharded_metadata = None
        index_file = None
        is_local = os.path.isdir(pretrained_model_name_or_path)
        index_file_kwargs = {
            "is_local": is_local,
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "subfolder": subfolder or "",
            "use_safetensors": use_safetensors,
            "cache_dir": cache_dir,
            "variant": variant,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "user_agent": user_agent,
            "commit_hash": commit_hash,
            "dduf_entries": dduf_entries,
        }
        index_file = _fetch_index_file(**index_file_kwargs)
        # In case the index file was not found we still have to consider the legacy format.
        # this becomes applicable when the variant is not None.
        if variant is not None and (index_file is None or not os.path.exists(index_file)):
            index_file = _fetch_index_file_legacy(**index_file_kwargs)
        if index_file is not None and (dduf_entries or index_file.is_file()):
            is_sharded = True

        # load model
        if from_flax:
            raise NotImplementedError("loading flax checkpoint in mindspore model is not yet supported.")
        else:
            # in the case it is sharded, we have already the index
            if is_sharded:
                resolved_model_file, sharded_metadata = _get_checkpoint_shard_files(
                    pretrained_model_name_or_path,
                    index_file,
                    cache_dir=cache_dir,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder or "",
                    dduf_entries=dduf_entries,
                )
            elif use_safetensors:
                try:
                    resolved_model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        commit_hash=commit_hash,
                        dduf_entries=dduf_entries,
                    )

                except IOError as e:
                    logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                    if not allow_pickle:
                        raise
                    logger.warning(
                        "Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead."
                    )

            if resolved_model_file is None and not is_sharded:
                resolved_model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                    dduf_entries=dduf_entries,
                )

        if not isinstance(resolved_model_file, list):
            resolved_model_file = [resolved_model_file]

        # set dtype to instantiate the model under:
        # 1. If mindspore_dtype is not None, we use that dtype
        # 2. If mindspore_dtype is float8, we don't use _set_default_mindspore_dtype and we downcast after loading the model
        dtype_orig = None  # noqa
        if mindspore_dtype is not None:
            if not isinstance(mindspore_dtype, ms.Type):
                raise ValueError(
                    f"{mindspore_dtype} needs to be of type `mindspore.Type`, e.g. `mindspore.float16`, but is {type(mindspore_dtype)}."
                )

        if mindspore_dtype is not None:
            patch_nn_default_dtype(dtype=mindspore_dtype, force=True)
        with no_init_parameters():
            model = cls.from_config(config, **unused_kwargs)
        if mindspore_dtype is not None:
            unpatch_nn_default_dtype()

        state_dict = None
        if not is_sharded:
            # Time to load the checkpoint
            state_dict = load_state_dict(resolved_model_file[0], disable_mmap=disable_mmap, dduf_entries=dduf_entries)
            # We only fix it for non sharded checkpoints as we don't need it yet for sharded one.
            model._fix_state_dict_keys_on_load(state_dict)

        if is_sharded:
            loaded_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            state_dict = _convert_state_dict(model, state_dict)
            loaded_keys = list(state_dict.keys())

        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = cls._load_pretrained_model(
            model,
            state_dict,
            resolved_model_file,
            pretrained_model_name_or_path,
            loaded_keys,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            dtype=mindspore_dtype,
            keep_in_fp32_modules=keep_in_fp32_modules,
            dduf_entries=dduf_entries,
            is_parallel_loading_enabled=is_parallel_loading_enabled,
        )
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
            "error_msgs": error_msgs,
        }

        if mindspore_dtype is not None:
            model = model.to(mindspore_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.set_train(False)

        if output_loading_info:
            return model, loading_info

        return model

    def to(self, dtype: Optional[ms.Type] = None):
        for p in self.get_parameters():
            if p.dtype != dtype:
                p._data = p.to(device="CPU", dtype=dtype)
        return self

    def half(self):
        for p in self.get_parameters():
            p._data = p.to(device="CPU", dtype=ms.float16)
        return self

    def float(self):
        for p in self.get_parameters():
            p._data = p.to(device="CPU", dtype=ms.float32)
        return self

    def compile_repeated_blocks(self, *args, **kwargs):
        """
        Compiles *only* the frequently repeated sub-modules of a model (e.g. the Transformer layers) instead of
        compiling the entire model. This technique—often called **regional compilation** (see the PyTorch recipe
        https://docs.pytorch.org/tutorials/recipes/regional_compilation.html) can reduce end-to-end compile time
        substantially, while preserving the runtime speed-ups you would expect from a full `torch.compile`.

        The set of sub-modules to compile is discovered by the presence of **`_repeated_blocks`** attribute in the
        model definition. Define this attribute on your model subclass as a list/tuple of class names (strings). Every
        module whose class name matches will be compiled.

        Once discovered, each matching sub-module is compiled by calling `submodule.compile(*args, **kwargs)`. Any
        positional or keyword arguments you supply to `compile_repeated_blocks` are forwarded verbatim to
        `torch.compile`.
        """
        repeated_blocks = getattr(self, "_repeated_blocks", None)

        if not repeated_blocks:
            raise ValueError(
                "`_repeated_blocks` attribute is empty. "
                f"Set `_repeated_blocks` for the class `{self.__class__.__name__}` to benefit from faster compilation. "
            )
        has_compiled_region = False
        for submod in self.cells():
            if submod.__class__.__name__ in repeated_blocks:
                submod.construct = ms.jit(submod.construct)
                has_compiled_region = True

        if not has_compiled_region:
            raise ValueError(
                f"Regional compilation failed because {repeated_blocks} classes are not found in the model. "
            )

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict: OrderedDict,
        resolved_model_file: List[str],
        pretrained_model_name_or_path: Union[str, os.PathLike],
        loaded_keys: List[str],
        ignore_mismatched_sizes: bool = False,
        dtype: Optional[Union[str, ms.Type]] = None,
        keep_in_fp32_modules: Optional[List[str]] = None,
        dduf_entries: Optional[Dict[str, DDUFEntry]] = None,
        is_parallel_loading_enabled: Optional[bool] = False,
    ):
        model_state_dict = {k: v for k, v in model.parameters_and_names()}
        expected_keys = list(model_state_dict.keys())
        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))
        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        mismatched_keys = []
        error_msgs = []

        offload_index = None
        state_dict_folder, state_dict_index = None, None
        if state_dict is not None:
            # load_state_dict will manage the case where we pass a dict instead of a file
            # if state dict is not None, it means that we don't need to read the files from resolved_model_file also
            resolved_model_file = [state_dict]

        # Prepare the loading function sharing the attributes shared between them.
        load_fn = functools.partial(
            _load_shard_files_with_threadpool if is_parallel_loading_enabled else _load_shard_file,
            model=model,
            model_state_dict=model_state_dict,
            dtype=dtype,
            keep_in_fp32_modules=keep_in_fp32_modules,
            dduf_entries=dduf_entries,
            loaded_keys=loaded_keys,
            unexpected_keys=unexpected_keys,
            offload_index=offload_index,
            state_dict_index=state_dict_index,
            state_dict_folder=state_dict_folder,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

        if is_parallel_loading_enabled:
            offload_index, state_dict_index, _mismatched_keys, _error_msgs = load_fn(resolved_model_file)
            error_msgs += _error_msgs
            mismatched_keys += _mismatched_keys
        else:
            shard_files = resolved_model_file
            if len(resolved_model_file) > 1:
                shard_files = logging.tqdm(resolved_model_file, desc="Loading checkpoint shards")

            for shard_file in shard_files:
                offload_index, state_dict_index, _mismatched_keys, _error_msgs = load_fn(shard_file)
                error_msgs += _error_msgs
                mismatched_keys += _mismatched_keys

        offload_index = None

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"  # noqa
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the"
                f" checkpoint was trained on, you can already use {model.__class__.__name__} for predictions"
                " without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be"
                " able to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        return expected_modules, optional_parameters

    # Adapted from `transformers` modeling_utils.py
    def _get_no_split_modules(self, device_map: str):
        """
        Get the modules of the model that should not be split when using device_map. We iterate through the modules to
        get the underlying `_no_split_modules`.

        Args:
            device_map (`str`):
                The device map value. Options are ["auto", "balanced", "balanced_low_0", "sequential"]

        Returns:
            `List[str]`: List of modules that should not be split
        """
        _no_split_modules = set()
        modules_to_check = [self]
        while len(modules_to_check) > 0:
            module = modules_to_check.pop(-1)
            # if the module does not appear in _no_split_modules, we also check the children
            if module.__class__.__name__ not in _no_split_modules:
                if isinstance(module, ModelMixin):
                    if module._no_split_modules is None:
                        raise ValueError(
                            f"{module.__class__.__name__} does not support `device_map='{device_map}'`. To implement support, the model "
                            "class needs to implement the `_no_split_modules` attribute."
                        )
                    else:
                        _no_split_modules = _no_split_modules | set(module._no_split_modules)
                modules_to_check += list(module.cells())
        return list(_no_split_modules)

    @classmethod
    def _set_default_mindspore_dtype(cls, dtype: ms.Type) -> ms.Type:
        """
        Change the default dtype and return the previous one. This is needed when wanting to instantiate the model
        under specific dtype.

        Args:
            dtype (`mindspore.Type`):
                a floating dtype to set to.

        Returns:
            `mindspore.Type`: the original `dtype` that can be used to restore `torch.set_default_dtype(dtype)` if it was
            modified. If it wasn't, returns `None`.

        Note `set_default_dtype` currently only works with floating-point types and asserts if for example,
        `ms.int64` is passed. So if a non-float `dtype` is passed this functions will throw an exception.
        """
        raise NotImplementedError("`_set_default_mindspore_dtype` is not yet supported.")

    @property
    def dtype(self) -> ms.Type:
        """
        `mindspore.Type`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (trainable or non-embedding) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters.
            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embedding parameters.

        Returns:
            `int`: The number of parameters.

        Example:

        ```py
        from mindone.diffusers import UNet2DConditionModel

        model_id = "runwayml/stable-diffusion-v1-5"
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        unet.num_parameters(only_trainable=True)
        859520964
        ```
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.cells_and_names()
                if isinstance(module_type, mint.nn.Embedding)
            ]
            total_parameters = [
                parameter for name, parameter in self.parameters_and_names() if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.get_parameters())

        total_numel = []

        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                # For 4bit models, we need to multiply the number of parameters by 2 as half of the parameters are
                # used for the 4bit quantization (uint8 tensors are stored)
                total_numel.append(param.numel())

        return sum(total_numel)

    def _set_gradient_checkpointing(self, enable: bool = True) -> None:
        is_gradient_checkpointing_set = False

        for name, module in self.cells_and_names():
            if hasattr(module, "recompute_"):
                logger.debug(f"Setting `gradient_checkpointing={enable}` for '{name}'")
                module.recompute_(enable)
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"The module {self.__class__.__name__} does not support gradient checkpointing. Please make sure to "
                f"use a module that supports gradient checkpointing by creating a boolean attribute `gradient_checkpointing`."
            )

    def _fix_state_dict_keys_on_load(self, state_dict: OrderedDict) -> None:
        """
        This function fix the state dict of the model to take into account some changes that were made in the model
        architecture:
        - deprecated attention blocks (happened before we introduced sharded checkpoint,
        so this is why we apply this method only when loading non sharded checkpoints for now)
        """
        deprecated_attention_block_paths = []

        def recursive_find_attn_block(name, module):
            if hasattr(module, "_from_deprecated_attn_block") and module._from_deprecated_attn_block:
                deprecated_attention_block_paths.append(name)

            for sub_name, sub_module in module.name_cells().items():
                sub_name = sub_name if name == "" else f"{name}.{sub_name}"
                recursive_find_attn_block(sub_name, sub_module)

        recursive_find_attn_block("", self)

        # NOTE: we have to check if the deprecated parameters are in the state dict
        # because it is possible we are loading from a state dict that was already
        # converted

        for path in deprecated_attention_block_paths:
            # group_norm path stays the same

            # query -> to_q
            if f"{path}.query.weight" in state_dict:
                state_dict[f"{path}.to_q.weight"] = state_dict.pop(f"{path}.query.weight")
            if f"{path}.query.bias" in state_dict:
                state_dict[f"{path}.to_q.bias"] = state_dict.pop(f"{path}.query.bias")

            # key -> to_k
            if f"{path}.key.weight" in state_dict:
                state_dict[f"{path}.to_k.weight"] = state_dict.pop(f"{path}.key.weight")
            if f"{path}.key.bias" in state_dict:
                state_dict[f"{path}.to_k.bias"] = state_dict.pop(f"{path}.key.bias")

            # value -> to_v
            if f"{path}.value.weight" in state_dict:
                state_dict[f"{path}.to_v.weight"] = state_dict.pop(f"{path}.value.weight")
            if f"{path}.value.bias" in state_dict:
                state_dict[f"{path}.to_v.bias"] = state_dict.pop(f"{path}.value.bias")

            # proj_attn -> to_out.0
            if f"{path}.proj_attn.weight" in state_dict:
                state_dict[f"{path}.to_out.0.weight"] = state_dict.pop(f"{path}.proj_attn.weight")
            if f"{path}.proj_attn.bias" in state_dict:
                state_dict[f"{path}.to_out.0.bias"] = state_dict.pop(f"{path}.proj_attn.bias")

        # TODO : MindSpore 2.6 share weight bug. Unable to load WTE and LM-Head layer weights properly. It will be
        #  deleted until fixed load_state_dict_into_model and parameters_and_names。
        if hasattr(self, "wte_lm_share") and self.wte_lm_share:
            state_dict["transformer.transformer.wte.embedding_table"] = state_dict["transformer.lm_head.weight"]

        return state_dict

    def get_submodule(self, target: str) -> nn.Cell:
        """Return the submodule given by ``target`` if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Cell`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Dense(input_channels=100, output_channels=200, has_bias=True)
                )
            )

        (The diagram shows an ``nn.Cell`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded by the degree
        of module nesting in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            nn.Cell: The submodule referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Cell``
        """
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: nn.Cell = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod.cls_name + " has no " "attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, nn.Cell):
                raise AttributeError("`" + item + "` is not " "an nn.Module")

        return mod


class LegacyModelMixin(ModelMixin):
    r"""
    A subclass of `ModelMixin` to resolve class mapping from legacy classes (like `Transformer2DModel`) to more
    pipeline-specific classes (like `DiTTransformer2DModel`).
    """

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # To prevent dependency import problem.
        from .model_loading_utils import _fetch_remapped_cls_from_config

        # Create a copy of the kwargs so that we don't mess with the keyword arguments in the downstream calls.
        kwargs_copy = kwargs.copy()

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, _, _ = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )
        # resolve remapping
        remapped_class = _fetch_remapped_cls_from_config(config, cls)

        if remapped_class is cls:
            return super(LegacyModelMixin, remapped_class).from_pretrained(pretrained_model_name_or_path, **kwargs_copy)
        else:
            return remapped_class.from_pretrained(pretrained_model_name_or_path, **kwargs_copy)
