import copy
import logging
import os
from typing import Dict, Optional, Union

from huggingface_hub import DDUFEntry
from huggingface_hub.utils import validate_hf_hub_args

import mindspore as ms
from mindspore.nn.utils import no_init_parameters

from mindone.diffusers import ModelMixin, __version__
from mindone.diffusers.configuration_utils import ConfigMixin
from mindone.diffusers.models.model_loading_utils import _fetch_index_file, _fetch_index_file_legacy, load_state_dict
from mindone.diffusers.models.modeling_utils import _convert_state_dict
from mindone.diffusers.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    _add_variant,
    _get_checkpoint_shard_files,
    _get_model_file,
)

logger = logging.getLogger(__name__)


class VideoBaseAE(ModelMixin, ConfigMixin):
    config_name = "config.json"
    _supports_gradient_checkpointing = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def download_and_load_model(cls, model_name, cache_dir=None):
        pass

    def encode(self, x: ms.Tensor, *args, **kwargs):
        pass

    def decode(self, encoding: ms.Tensor, *args, **kwargs):
        pass

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # adapted from mindone.diffusers.models.modeling_utils.from_pretrained
        state_dict = kwargs.pop("state_dict", None)  # additional key argument
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
        use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (mindspore_dtype == ms.float16)

        if use_keep_in_fp32_modules:
            keep_in_fp32_modules = cls._keep_in_fp32_modules
            if not isinstance(keep_in_fp32_modules, list):
                keep_in_fp32_modules = [keep_in_fp32_modules]
        else:
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

        with no_init_parameters():
            model = cls.from_config(config, **unused_kwargs)

        # state_dict = None # state_dict may be passed as an additional key argument
        if state_dict is None:  # edits: only load model_file if state_dict is None
            if not is_sharded:
                # Time to load the checkpoint
                state_dict = load_state_dict(
                    resolved_model_file[0], disable_mmap=disable_mmap, dduf_entries=dduf_entries
                )
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
        )
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
            "error_msgs": error_msgs,
        }

        if mindspore_dtype is not None and not use_keep_in_fp32_modules:
            model = model.to(mindspore_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.set_train(False)

        if output_loading_info:
            return model, loading_info

        return model
