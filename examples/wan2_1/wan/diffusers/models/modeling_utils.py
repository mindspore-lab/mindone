import os
from typing import Optional, OrderedDict, Union

import mindspore as ms

from mindone.diffusers.models.modeling_utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    ModelMixin,
    __version__,
    _add_variant,
    _fetch_index_file,
    _get_checkpoint_shard_files,
    _get_model_file,
    logger,
)

from .model_loading_utils import load_checkpoint_and_dispatch, load_state_dict


class ModelMixinFastLoad(ModelMixin):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
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

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

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
            **kwargs,
        )

        # Determine if we're loading from a directory of sharded checkpoints.
        is_sharded = False
        index_file = None
        is_local = os.path.isdir(pretrained_model_name_or_path)
        index_file = _fetch_index_file(
            is_local=is_local,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder or "",
            use_safetensors=use_safetensors,
            cache_dir=cache_dir,
            variant=variant,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            user_agent=user_agent,
            commit_hash=commit_hash,
        )
        if index_file is not None and index_file.is_file():
            is_sharded = True

        # load model
        model_file = None
        if from_flax:
            raise NotImplementedError("loading flax checkpoint in mindspore model is not yet supported.")
        else:
            if is_sharded:
                sharded_ckpt_cached_folder, sharded_metadata = _get_checkpoint_shard_files(
                    pretrained_model_name_or_path,
                    index_file,
                    cache_dir=cache_dir,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder or "",
                )

            elif use_safetensors and not is_sharded:
                try:
                    model_file = _get_model_file(
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
                    )
                except IOError as e:
                    logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                    if not allow_pickle:
                        raise
                    logger.warning(
                        "Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead."
                    )

            if model_file is None and not is_sharded:
                model_file = _get_model_file(
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
                )

            # Move the model's data type conversion ahead of the weight loading process to avoid unnecessary
            # data type conversions of weights that can increase computation time in certain situations.
            if mindspore_dtype is not None and not isinstance(mindspore_dtype, ms.Type):
                raise ValueError(
                    f"{mindspore_dtype} needs to be of type `ms.Type`, e.g. `ms.float16`, but is {type(mindspore_dtype)}."
                )
            elif mindspore_dtype is not None:
                config["dtype"] = mindspore_dtype

            model = cls.from_config(config, **unused_kwargs)

            if is_sharded:
                load_checkpoint_and_dispatch(
                    model,
                    index_file,  # TODO: check accelerate
                    dtype=config.get("dtype", ms.float32),
                    strict=True,
                )
            else:
                state_dict = load_state_dict(model_file, variant=variant, dtype=config.get("dtype", ms.float32))
                model._convert_deprecated_attention_blocks(state_dict)

                model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                    model,
                    state_dict,
                    model_file,
                    pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                )

                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.set_train(False)
        if not is_sharded and output_loading_info:
            return model, loading_info

        return model

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict: OrderedDict,
        resolved_archive_file,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        ignore_mismatched_sizes: bool = False,
    ):
        # Retrieve missing & unexpected_keys
        model_state_dict = {k: v for k, v in model.parameters_and_names()}
        loaded_keys = list(state_dict.keys())

        expected_keys = list(model_state_dict.keys())

        original_loaded_keys = loaded_keys

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Make sure we are able to load base models as well as derived models (with heads)
        model_to_load = model

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes,
            )
            ms.load_param_into_net(model_to_load, state_dict, strict_load=True)

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task"
                " or with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly"
                " identical (initializing a BertForSequenceClassification model from a"
                " BertForSequenceClassification model)."
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

        return model, missing_keys, unexpected_keys, mismatched_keys, None
