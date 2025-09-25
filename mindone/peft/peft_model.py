# Copyright 2023-present the HuggingFace Inc. team.
#
# This code is adapted from https://github.com/huggingface/peft
# with modifications to run peft on mindspore.
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

from __future__ import annotations

import collections
import copy
import inspect
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import packaging.version
from huggingface_hub import ModelCard, ModelCardData, hf_hub_download
from transformers.utils import PushToHubMixin

import mindspore as ms
from mindspore import mint, nn

from mindone.peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from mindone.peft.utils.constants import DUMMY_MODEL_CONFIG
from mindone.safetensors.mindspore import save_file as safe_save_file
from mindone.transformers import MSPreTrainedModel as PreTrainedModel
from mindone.transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache

from . import __version__
from .config import PeftConfig
from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_PREFIX_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING
from .utils import (
    SAFETENSORS_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftType,
    TaskType,
    _get_batch_size,
    _get_subcell,
    _prepare_prompt_learning_config,
    _set_adapter,
    _set_trainable,
    get_peft_model_state_dict,
    load_peft_weights,
    refresh_parameter_name_of_model,
    set_peft_model_state_dict,
)


class PeftModel(PushToHubMixin, nn.Cell):
    """
    Base model encompassing various Peft methods.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    **Attributes**:
        - **base_model** ([`nn.Cell`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
            saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
            using [`PromptLearningConfig`].
        - **prompt_tokens** (`ms.Tensor`) -- The virtual prompt tokens used for Peft if
            using [`PromptLearningConfig`].
        - **transformer_backbone_name** (`str`) -- The name of the transformer
            backbone in the base model if using [`PromptLearningConfig`].
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
            in the base model if using [`PromptLearningConfig`].
    """

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.modules_to_save = None
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        # These args are special PEFT arguments that users can pass. They need to be removed before passing them to
        # forward.
        self.special_peft_forward_args = {"adapter_names"}

        self._is_prompt_learning = peft_config.is_prompt_learning
        if self._is_prompt_learning:
            self._peft_config = {adapter_name: peft_config}
            self.base_model = model
            self.add_adapter(adapter_name, peft_config)
        else:
            self._peft_config = None
            cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
            self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)

        self.set_additional_trainable_modules(peft_config, adapter_name)

        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1

    @property
    def peft_config(self) -> dict[str, PeftConfig]:
        if self._is_prompt_learning:
            return self._peft_config
        return self.base_model.peft_config

    @property
    def active_adapters(self) -> list[str]:
        try:
            adapters = self.base_model.active_adapters
            if not isinstance(adapters, list):
                # Base model is probably a transformers model, see:
                # https://github.com/huggingface/transformers/pull/30790#issuecomment-2253808249
                # Unfortunately, transformers models also have an active_adapters method but it's 1) not a property and
                # 2) calling it fails because the base model (usually) has no loaded adapter. The base model can be a
                # transformers model for prompt learning, where the base model is not wrapped in a LoraModel or similar.
                adapters = self.active_adapter
                if isinstance(adapters, str):
                    adapters = [adapters]
        except AttributeError:
            adapters = self.active_adapter
            if isinstance(adapters, str):
                adapters = [adapters]
        return adapters

    @peft_config.setter
    def peft_config(self, value: dict[str, PeftConfig]):
        if self._is_prompt_learning:
            self._peft_config = value
        else:
            self.base_model.peft_config = value

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[list[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        path_initial_model_for_weight_conversion: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`PeftModel.from_pretrained`] class method, and also used by the [`PeftModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            safe_serialization (`bool`, *optional*):
                Whether to save the adapter files in safetensors format, defaults to `True`.
            selected_adapters (`List[str]`,  *optional*):
                A list of adapters to be saved. If `None`, will default to all adapters.
            save_embedding_layers (`Union[bool, str]`, *optional*, defaults to `"auto"`):
                If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common
                embedding layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available.
                and automatically sets the boolean flag. This only works for 🤗 transformers models.
            is_main_process (`bool`, *optional*):
                Whether the process calling this is the main process or not. Will default to `True`. Will not save the
                checkpoint if not on the main process, which is important for multi device setups (e.g. DDP).
            path_initial_model_for_weight_conversion (`str, *optional*`):
                The path to the initialized adapter, which is obtained after initializing the model with
                PiSSA/CorDA/OLoRA and before performing any training. When `path_initial_model_for_weight_conversion`
                is not None, the difference in adapter before and after fine-tuning is calculated. This difference can
                be represented as the parameters of a standard LoRA adapter. Using this converted adapter does not
                require changes to the base model, thus conveniently allowing the use of multiple PiSSA/CorDA/OLoRA
                adapters with LoRA adapters, and the activation or deactivation of any adapters. Note that this
                conversion is not supported if `rslora` is used in combination with `rank_pattern` or `alpha_pattern`.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.

        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        def save_mutated_as_lora(peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs):
            if peft_config.use_rslora and (peft_config.rank_pattern or peft_config.alpha_pattern):
                msg = (
                    "Passing `path_initial_model_for_weight_conversion` to `save_pretrained` is not supported when "
                    "using `rank_pattern` or `alpha_pattern` at the same time as `use_rslora=True`."
                )
                raise ValueError(msg)

            if not any(
                str(peft_config.init_lora_weights).lower().startswith(prefix)
                for prefix in ["pissa", "corda", "olora", "true"]
            ):
                warnings.warn(
                    "`path_initial_model_for_weight_conversion` only works for converting a PiSSA/CorDA/OLoRA adapter to "
                    "a LoRA adapter"
                )
            initial_adapter_name = os.path.basename(path_initial_model_for_weight_conversion)
            try:
                self.load_adapter(
                    os.path.dirname(path_initial_model_for_weight_conversion),
                    subfolder=initial_adapter_name,
                    adapter_name=initial_adapter_name,
                )
                is_pissa = str(self.peft_config[initial_adapter_name].init_lora_weights).lower().startswith("pissa")
                is_corda = str(self.peft_config[initial_adapter_name].init_lora_weights).lower() == "corda"
                is_olora = str(self.peft_config[initial_adapter_name].init_lora_weights).lower() == "olora"
                if is_pissa or is_corda or is_olora:
                    raise ValueError(
                        "The `init_lora_weights` parameter of the initial adapter should be set to `True`. "
                        "Otherwise, `self.load_adapter` will subtract the decomposed values again based on the "
                        "residual model."
                    )
                output_state_dict = self.base_model.subtract_mutated_init(
                    output_state_dict, initial_adapter_name, kwargs
                )
            finally:
                self.delete_adapter(initial_adapter_name)
            return output_state_dict

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            self.create_or_update_model_card(save_directory)

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
                save_embedding_layers=save_embedding_layers,
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if is_main_process and safe_serialization:
                # Original PEFT:
                #   Safetensors does not allow tensor aliasing. We're going to remove aliases before saving
                #   Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
                # MindSpore does not have tensor aliasing. We don't need to remove aliases before saving

                if path_initial_model_for_weight_conversion is not None:
                    peft_config = copy.deepcopy(peft_config)
                    peft_config.init_lora_weights = True
                    peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                    output_state_dict = save_mutated_as_lora(
                        peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs
                    )
                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "np"},
                )
            elif is_main_process:
                if path_initial_model_for_weight_conversion is not None:
                    peft_config = copy.deepcopy(peft_config)
                    peft_config.init_lora_weights = True
                    peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                    output_state_dict = save_mutated_as_lora(
                        peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs
                    )
                ms.save_checkpoint(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if peft_config.is_prompt_learning
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            if peft_config.task_type is None:
                # deal with auto mapping
                base_model_class = self._get_base_model_class(
                    is_prompt_tuning=peft_config.is_prompt_learning,
                )
                parent_library = base_model_class.__module__

                auto_mapping_dict = {
                    "base_model_class": base_model_class.__name__,
                    "parent_library": parent_library,
                }
            else:
                auto_mapping_dict = None

            if is_main_process:
                if path_initial_model_for_weight_conversion is not None:
                    peft_config.init_lora_weights = True
                    peft_config.r *= 2
                    if not peft_config.use_rslora:
                        peft_config.lora_alpha *= 2
                    else:
                        # with rslora, we have scaling = alpha / sqrt(r), we thus adjust alpha to keep the same scaling
                        peft_config.lora_alpha *= 2**0.5

                    if peft_config.rank_pattern:
                        peft_config.rank_pattern = {key: 2 * val for key, val in peft_config.rank_pattern.items()}
                    if peft_config.alpha_pattern:
                        peft_config.alpha_pattern = {key: 2 * val for key, val in peft_config.alpha_pattern.items()}

                peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Cell,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        **kwargs: Any,
    ) -> PeftModel:
        r"""
        Instantiate a PEFT model from a pretrained model and loaded PEFT weights.

        Note that the passed `model` may be modified inplace.

        Args:
            model ([`nn.Cell`]):
                The model to be adapted. For 🤗 Transformers models, the model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`].
            model_id (`str` or `os.PathLike`):
                The name of the PEFT configuration to use. Can be either:
                    - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                      method (`./my_peft_config_directory/`).
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to be loaded. This is useful for loading multiple adapters.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            config ([`~peft.PeftConfig`], *optional*):
                The configuration object to use instead of an automatically loaded configuration. This configuration
                object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
                loaded before calling `from_pretrained`.
            autocast_adapter_dtype (`bool`, *optional*):
                Whether to autocast the adapter dtype. Defaults to `True`. Only relevant for specific adapter types.
            ephemeral_gpu_offload (`bool`, *optional*):
                Whether to use ephemeral GPU offloading for partially loaded modules. Defaults to `False`. This is
                useful when parts of the model and/or components (such as adapters) are kept in CPU memory until they
                are needed. Rather than perform expensive operations on small data, the data is transferred to the GPU
                on-demand, the operation(s) performed, and the results moved back to CPU memory. This brings a slight
                momentary VRAM overhead but gives orders of magnitude speedup in certain cases.
            kwargs: (`optional`):
                Additional keyword arguments passed along to the specific PEFT configuration class.
        """
        from .auto import MODEL_TYPE_TO_PEFT_MODEL_MAPPING

        # load the config
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                    use_auth_token=kwargs.get("use_auth_token", None),
                    token=kwargs.get("token", None),
                )
            ].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        if config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config, adapter_name, autocast_adapter_dtype=autocast_adapter_dtype)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](
                model,
                config,
                adapter_name,
                autocast_adapter_dtype=autocast_adapter_dtype,
            )

        load_result = model.load_adapter(
            model_id,
            adapter_name,
            is_trainable=is_trainable,
            autocast_adapter_dtype=autocast_adapter_dtype,
            **kwargs,
        )

        # 1. Remove VB-LoRA vector bank, since it's a shared parameter set via the VBLoRAModel
        # 2. Remove the prompt encoder, as it does not need to be part of the checkpoint
        missing_keys = [
            k for k in load_result["missing_keys"] if "vblora_vector_bank" not in k and "prompt_encoder" not in k
        ]
        if missing_keys:
            # Let's warn here since (in contrast to load_adapter) we don't return the load result, so it could be quite
            # difficult for users to even notice that something might have gone wrong here. As we filter out non PEFT
            # keys from the missing keys, this gives no false positives.

            warn_message = f"Found missing adapter keys while loading the checkpoint: {missing_keys}."

            prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(config.peft_type)
            if prefix and adapter_name in prefix:
                warn_message += (
                    f"Adapter name {adapter_name} should not be contained in the prefix {prefix}."
                    "This could be the potential reason for missing adapter keys."
                )

            warnings.warn(warn_message)

        return model

    def _setup_prompt_encoder(self, adapter_name: str):
        config = self.peft_config[adapter_name]
        if not hasattr(self, "prompt_encoder"):
            self.prompt_encoder = nn.CellDict({})
            self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.name_cells():
            for param in module.get_parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
        if transformer_backbone is None:
            transformer_backbone = self.base_model

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        # determine the word embeddings
        word_embeddings = None
        try:
            # First try to find the word embeddings based on the module name, this should work for models like Bert,
            # Roberta, Deberta, etc.
            word_embeddings = _get_subcell(self.base_model, "embeddings.word_embeddings")
        except AttributeError:
            pass

        if word_embeddings is None:
            # Word embeddings could not be determined. Next try to guess them by checking which parameter has the size
            # of the vocab.
            for named_param, value in list(transformer_backbone.parameters_and_names()):
                if value.shape[0] == self.base_model.config.vocab_size:
                    word_embeddings = _get_subcell(transformer_backbone, named_param.replace(".weight", ""))
                    break

        self.word_embeddings = word_embeddings
        model_cls = PEFT_TYPE_TO_TUNER_MAPPING[config.peft_type]

        if config.peft_type in (PeftType.PROMPT_TUNING, PeftType.MULTITASK_PROMPT_TUNING, PeftType.CPT):
            prompt_encoder = model_cls(config, self.word_embeddings)
        elif config.peft_type == PeftType.P_TUNING:
            prompt_encoder = model_cls(config)
        elif config.peft_type == PeftType.PREFIX_TUNING:
            # prefix tuning now uses Cache but that won't work with gradient checkpointing
            if any(getattr(module, "gradient_checkpointing", False) for module in self.get_base_model().cells()):
                raise ValueError("Prefix tuning does not work with gradient checkpointing.")
            prompt_encoder = model_cls(config)
        else:
            raise ValueError("Not supported")

        self.prompt_encoder.update(nn.CellDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = mint.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).to(ms.int64)

    def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        r"""
        Prepares the model for gradient checkpointing if necessary
        """
        # Since MindSpore and PyTorch have different machanisms for autograd, we don't have to do `enable_input_require_grads`
        # or `make_inputs_require_grad` as PyTorch does. Instead, we can just return the model as is.
        return model

    def get_prompt_embedding_to_save(self, adapter_name: str) -> ms.Tensor:
        """
        Returns the prompt embedding to save when saving the model. Only applicable when using a prompt learning
        method.
        """
        raise NotImplementedError

    def get_prompt(self, batch_size: int, task_ids: Optional[ms.Tensor] = None) -> ms.Tensor:
        """
        Returns the virtual prompts to use for Peft. Only applicable when using a prompt learning method.
        """
        peft_config = self.active_peft_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = self.prompt_tokens[self.active_adapter].unsqueeze(0).expand(batch_size, -1)
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
            if peft_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = prompt_encoder(prompt_tokens)
            if self.base_model_torch_dtype is not None:
                past_key_values = past_key_values.to(self.base_model_torch_dtype)
            past_key_values = past_key_values.view(
                batch_size,
                peft_config.num_virtual_tokens,
                peft_config.num_layers * 2,
                peft_config.num_attention_heads,
                peft_config.token_dim // peft_config.num_attention_heads,
            )
            if peft_config.num_transformer_submodules == 2:
                past_key_values = mint.cat([past_key_values, past_key_values], dim=2)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(peft_config.num_transformer_submodules * 2)
            if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
                post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
                past_key_values = post_process_fn(past_key_values)
            elif peft_config.num_transformer_submodules == 1:
                # Dont' apply this to encoder-decoder models and not to models requiring special processing.
                # local import in case users use a very old transformers version
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            elif peft_config.num_transformer_submodules == 2 and self.base_model._supports_cache_class:
                # Dont' apply this to encoder-decoder models that don't support new Cachc format yet
                # If we don't apply this, prefix-tuning fails to update cross-attn cache
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
                past_key_values.cross_attention_cache = DynamicCache()
                past_key_values.is_updated = {
                    layer_idx: False for layer_idx in range(len(past_key_values.cross_attention_cache.key_cache))
                }
            return past_key_values
        else:
            if peft_config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
                prompts = prompt_encoder(prompt_tokens, task_ids)
            else:
                if peft_config.inference_mode:
                    prompts = prompt_encoder.embedding.weight
                else:
                    # Take only one prompt token sample and expand the output instead of expanding the input, see:
                    # https://github.com/huggingface/peft/issues/2043#issuecomment-2321522577
                    prompt_tokens = prompt_tokens[:1]
                    prompts = prompt_encoder(prompt_tokens)
                prompts = prompts.repeat(batch_size, 1, 1)
            return prompts

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.parameters_and_names():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.

        Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
        num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
        (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
        For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
        prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
        of trainable parameters of the backbone transformer model which can be different.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "base_model":  # prevent infinite recursion if class is not initialized
                raise
            return getattr(self.base_model, name)

    def construct(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
        return self.get_base_model()(*args, **kwargs)

    def generate(self, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
        return self.get_base_model().generate(*args, **kwargs)

    def _get_base_model_class(self, is_prompt_tuning=False):
        """
        Returns the base model class.
        """
        if not is_prompt_tuning:
            return self.base_model.model.__class__
        return self.base_model.__class__

    @contextmanager
    def disable_adapter(self):
        """
        Context manager that disables the adapter module. Use this to run inference on the base model.

        Example:

        ```py
        >>> with model.disable_adapter():
        ...     model(inputs)
        ```
        """
        if self.peft_config[self.active_adapter].is_prompt_learning:
            try:
                # TODO: consider replacing this patching of methods with a more robust mechanism: setting a flag and
                # letting the underlying methods deal with it, same as how LoRA does it.
                old_construct = self.construct
                self.construct = self.base_model.construct
                old_prepare_inputs_for_generation = self.prepare_inputs_for_generation
                self.prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
                yield
            finally:
                self.construct = old_construct
                self.prepare_inputs_for_generation = old_prepare_inputs_for_generation

        elif self.peft_config[self.active_adapter].is_adaption_prompt:
            try:
                self.base_model.disable_adapter_layers()
                yield
            finally:
                self.base_model.enable_adapter_layers()

        else:  # LoRA, LoHa, etc.
            model_status = self.get_model_status()
            if model_status.enabled == "irregular":
                warnings.warn(
                    "The model contains some adapter layers that are enabled and others that are disabled. "
                    "This is most likely unintentional. After exiting the disable_adapter context, all adapters "
                    "will be enabled"
                )
            try:
                self.base_model.disable_adapter_layers()
                yield
            finally:
                if model_status.enabled is not False:
                    # model_status.enabled is `True` or `"irregular"`
                    self.base_model.enable_adapter_layers()

    def get_base_model(self) -> nn.Cell:
        """
        Returns the base model.
        """
        return (
            self.base_model
            if (self.active_peft_config.is_prompt_learning or self.peft_type == PeftType.POLY)
            else self.base_model.model
        )

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig, **kwargs) -> None:
        """
        Add an adapter to the model based on the passed configuration.

        This adapter is not trained. To load a trained adapter, check out [`PeftModel.load_adapter`].

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
        """
        prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(peft_config.peft_type)
        if prefix and adapter_name in prefix:
            warnings.warn(
                f"Adapter name {adapter_name} should not be contained in the prefix {prefix}."
                "This may lead to reinitialization of the adapter weights during loading."
            )

        if peft_config.peft_type != self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}."
            )

        try:
            if peft_config.is_prompt_learning:
                self.peft_config[adapter_name] = peft_config
                if hasattr(self.config, "to_dict"):
                    dict_config = self.config.to_dict()
                else:
                    dict_config = self.config

                peft_config = _prepare_prompt_learning_config(peft_config, dict_config)
                self._setup_prompt_encoder(adapter_name)
            elif peft_config.is_adaption_prompt:
                self.base_model.add_adapter(adapter_name, peft_config)
            else:
                self.peft_config[adapter_name] = peft_config
                self.base_model.inject_adapter(self.base_model.model, adapter_name)
        except Exception:  # something went wrong, roll back
            if adapter_name in self.peft_config:
                del self.peft_config[adapter_name]
            raise

        self.set_additional_trainable_modules(peft_config, adapter_name)

    def set_additional_trainable_modules(self, peft_config, adapter_name):
        if getattr(peft_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(peft_config.modules_to_save)
            else:
                self.modules_to_save.update(peft_config.modules_to_save)
            # this may add a new ModulesToSaveWrapper
            _set_trainable(self, adapter_name, module_names=getattr(peft_config, "modules_to_save", None))

        if getattr(peft_config, "trainable_token_indices", None) is not None:
            raise NotImplementedError

    def get_layer_status(self) -> list[TunerLayerStatus]:
        """Get the status of each adapter layer in the model.

        This method returns a list of `TunerLayerStatus` dataclass instances, each of which contains the following
        attributes:

        - `name` (`str`):
           The name of the adapter layer, e.g. `model.encoder.block.0.layer.0.SelfAttention.q`.
        - `module_type` (`str`):
           The type of the adapter layer, e.g. `lora.Linear`.
        - `enabled` (`bool`):
           Whether the adapter layer is enabled.
        - `active_adapters` (`list[str]`):
           The names of the active adapters, if any, e.g. `["default"]`.
        - `merged_adapters` (`list[str]`):
           The names of the merged adapters, if any, e.g. `["default"]`.
        - `available_adapters` (`list[str]`):
           The names of the available adapters, e.g. `["default"]`.

        Args:
            model ([`~PeftModel`]):
                The model to get the adapter layer status from.

        Returns:
            list[`peft.peft_model.TunerLayerStatus`]:
                A list of dataclasses, each containing the status of the corresponding adapter layer.

        """
        return get_layer_status(self)

    def get_model_status(self) -> TunerModelStatus:
        """Get the status of tuners of the model.

        This method returns a `TunerModelStatus` dataclass instance, which contains the following attributes:

        - `base_model_type` (`str`):
           The type of the base model, e.g. `T5Model`.
        - `adapter_model_type` (`str`):
           The type of the adapter model, e.g. `LoraModel`.
        - `peft_types` (`dict[str, str]`):
           The mapping of adapter name to adapter type, e.g. `{"default": "LORA"}`.
        - `trainable_params` (`int`):
           The number of trainable parameters in the model.
        - `total_params` (`int`):
           The total number of parameters in the model.
        - `num_adapter_layers` (`int`):
           The number of adapter layers in the model.
        - `enabled` (`bool`, `Literal["irregular"]`):
           Whether all adapter layers are enabled. If some are enabled and some are not, this will be `"irregular"`.
           This means that your model is in an inconsistent state and might not work as expected.
        - `active_adapters` (`list[str]`, `Literal["irregular"]`):
           The names of the active adapters. If the active adapters are not consistent across all layers, this will be
           `"irregular"`, which means that your model is in an inconsistent state and might not work as expected.
        - `merged_adapters` (`list[str]`, `Literal["irregular"]`):
           The names of the merged adapters. If the merged adapters are not consistent across all layers, this will be
           `"irregular"`, which means that your model is in an inconsistent state and might not work as expected.
        - `available_adapters` (`list[str]`):
           The names of the available adapters, e.g. `["default"]`.

        Args:
            model ([`~PeftModel`]):
                The model to get the adapter layer status from.

        Returns:
            `peft.peft_model.TunerModelStatus`:
                A dataclass containing the status of the model.

        """
        return get_model_status(self)

    @classmethod
    def _split_kwargs(cls, kwargs: dict[str, Any]):
        _kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
        hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters or key in _kwargs_not_in_hf_hub_download_signature:
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, other_kwargs

    def _update_offload(self, offload_index: dict[str, dict[str, str]], adapters_weights: dict[str, ms.Tensor]):
        """
        Update the offload_index and safetensors files for loading and mergine PeftModels with disk-offloaded modules.

        Args:
            offload_index (Dict[str: str]):
                Dictionary of disk-offloaded modules with their metadata and safetensors filenames
            adapters_weights (Dict[str: ms.Tensor]):
                Dictionary of Peft adapter module names and weights
        """

        raise NotImplementedError("Offload is not supported.")

    def _check_new_adapter_config(self, peft_config: PeftConfig, is_trainable: bool) -> None:
        """Perform checks on newly added PEFT configs to ensure integrity."""
        if peft_config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")

        # Since PiSSA/CorDA/OLoRA modifies the base weights, it should not be combined with other adapters.
        all_configs = [peft_config] + list(self.peft_config.values())
        if len(all_configs) > 1:
            if any(getattr(config, "init_lora_weights", None) == "pissa" for config in all_configs):
                msg = (
                    "PiSSA changes the base weights of the model and should thus not be used with other adapters. "
                    "Consider converting the PiSSA adapter into a normal LoRA adapter: "
                    "https://github.com/huggingface/peft/tree/main/examples/pissa_finetuning#convert-pissa-to-lora"
                )
                warnings.warn(msg)
            elif any(getattr(config, "init_lora_weights", None) == "corda" for config in all_configs):
                msg = (
                    "CorDA changes the base weights of the model and should thus not be used with other adapters. "
                    "Consider converting the CorDA adapter into a normal LoRA adapter: "
                    "https://github.com/huggingface/peft/tree/main/examples/corda_finetuning#convert-corda-to-lora"
                )
                warnings.warn(msg)
            elif any(getattr(config, "init_lora_weights", None) == "olora" for config in all_configs):
                msg = (
                    "OLoRA changes the base weights of the model and should thus not be used with other adapters. "
                    "Consider converting the OLoRA adapter into a normal LoRA adapter: "
                    "https://github.com/huggingface/peft/tree/main/examples/olora_finetuning#olora-and-lora"
                )
                warnings.warn(msg)

    def load_adapter(
        self,
        model_id: Union[str, os.PathLike],
        adapter_name: str,
        is_trainable: bool = False,
        autocast_adapter_dtype: bool = True,
        **kwargs: Any,
    ):
        """
        Load a trained adapter into the model.

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            model_id (`str` or `os.PathLike`):
                The name of the PEFT configuration to use. Can be either:
                    - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                      method (`./my_peft_config_directory/`).
            adapter_name (`str`):
                The name of the adapter to be added.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            autocast_adapter_dtype (`bool`, *optional*, defaults to `True`):
                Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter
                weights using float16 and bfloat16 to float32, as this is typically required for stable training, and
                only affect select PEFT tuners.
            kwargs: (`optional`):
                Additional arguments to modify the way the adapter is loaded, e.g. the token for Hugging Face Hub.
        """
        from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)

        if adapter_name not in self.peft_config:
            # load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    **hf_hub_download_kwargs,
                )
            ].from_pretrained(
                model_id,
                **hf_hub_download_kwargs,
            )
            self._check_new_adapter_config(peft_config, is_trainable=is_trainable)
            peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config)
            refresh_parameter_name_of_model(self, only_peft=True)  # Only MindSpore can do

        adapters_weights = load_peft_weights(model_id, **hf_hub_download_kwargs)

        # load the weights into the model
        ignore_mismatched_sizes = kwargs.get("ignore_mismatched_sizes", False)
        load_result = set_peft_model_state_dict(
            self,
            adapters_weights,
            adapter_name=adapter_name,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

        tuner = self.peft_config[adapter_name].peft_type
        tuner_prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(tuner, "")
        adapter_missing_keys = []

        # Filter missing keys specific to the current adapter and tuner prefix.
        for key in load_result["missing_keys"]:
            if tuner_prefix in key and adapter_name in key:
                adapter_missing_keys.append(key)

        load_result["missing_keys"].clear()
        load_result["missing_keys"].extend(adapter_missing_keys)

        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            raise NotImplementedError(f"hf_device_map is not supported, but got {self.hf_device_map = }")

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.set_train(False)
        return load_result

    def set_adapter(self, adapter_name: str) -> None:
        """
        Sets the active adapter.

        Only one adapter can be active at a time.

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str`):
                The name of the adapter to be set as active. The adapter must be loaded first.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        if not self.peft_config[adapter_name].is_prompt_learning:
            self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    @property
    def base_model_torch_dtype(self):
        return getattr(self.base_model, "dtype", None)

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]

    def create_or_update_model_card(self, output_dir: str):
        """
        Updates or create model card to include information about peft:
        1. Adds `peft` library tag
        2. Adds peft version
        3. Adds base model info
        4. Adds quantization information if it was used
        """

        filename = os.path.join(output_dir, "README.md")

        card = ModelCard.load(filename) if os.path.exists(filename) else ModelCard.from_template(ModelCardData())

        card.data["library_name"] = "peft"

        model_config = BaseTuner.get_model_config(self)
        model_config = None if model_config == DUMMY_MODEL_CONFIG else model_config
        if model_config is not None and "_name_or_path" in model_config:
            card.data["base_model"] = model_config["_name_or_path"]

        lines = card.text.splitlines()

        quantization_config = None
        if hasattr(model_config, "quantization_config"):
            quantization_config = self.config.quantization_config.to_dict()
        training_config_text = ""
        quantization_prefix = "The following `bitsandbytes` quantization config was used during training:"
        # Adds quantization information if it was used
        if quantization_config is not None:
            training_config_text += f"\n{quantization_prefix}\n"
            training_config_text += "\n".join([f"- {name}: {value}" for name, value in quantization_config.items()])
            training_config_text += "\n"

        training_procedure_heading = "## Training procedure"
        if quantization_prefix not in lines and bool(training_config_text):
            if training_procedure_heading in lines:
                lines.insert(lines.index(training_procedure_heading) + 2, training_config_text)
            else:
                lines.append(f"{training_procedure_heading}\n{training_config_text}")

        # Adds peft version
        framework_block_heading = "### Framework versions"
        if f"- PEFT {__version__}" not in lines:
            if framework_block_heading in lines:
                lines.insert(lines.index(framework_block_heading) + 2, f"- PEFT {__version__}")
            else:
                lines.append(f"{framework_block_heading}\n\n- PEFT {__version__}")

        card.text = "\n".join(lines)
        card.save(filename)

    # Copied from mindone.diffusers.models.modeling_utils.ModelMixin.to
    def to(self, dtype: Optional[ms.Type] = None):
        for p in self.get_parameters():
            p.set_dtype(dtype)
        return self

    # Copied from mindone.diffusers.models.modeling_utils.ModelMixin.half
    def half(self):
        for p in self.get_parameters():
            p.set_dtype(ms.float16)
        return self

    # Copied from mindone.diffusers.models.modeling_utils.ModelMixin.float
    def float(self):
        for p in self.get_parameters():
            p.set_dtype(ms.float32)
        return self


class PeftModelForCausalLM(PeftModel):
    """
    Peft model for causal language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    Example:

        ```py
        >>> from mindone.transformers import AutoModelForCausalLM
        >>> from mindone.peft import PeftModelForCausalLM, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "CAUSAL_LM",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 1280,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 20,
        ...     "num_layers": 36,
        ...     "encoder_hidden_size": 1280,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        >>> peft_model = PeftModelForCausalLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
        ```
    """

    def __init__(self, model: nn.Cell, peft_config: PeftConfig, adapter_name: str = "default", **kwargs) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = mint.ones(batch_size, peft_config.num_virtual_tokens)
            attention_mask = mint.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            # overwrite past_kv in kwargs
            kwargs["past_key_values"] = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        elif peft_config.peft_type == PeftType.CPT:
            return self._cpt_forward(input_ids, inputs_embeds, peft_config, task_ids, batch_size, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = mint.full((batch_size, peft_config.num_virtual_tokens), -100)
                kwargs["labels"] = mint.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = mint.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _cpt_forward(self, input_ids, inputs_embeds, peft_config, task_ids, batch_size, **kwargs):
        raise NotImplementedError

    def generate(self, *args, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            if not peft_config.is_prompt_learning:
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                outputs = self.base_model.generate(*args, **kwargs)
            else:
                outputs = self.base_model.generate(**kwargs)
        except:  # noqa: E722
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: Optional[ms.Tensor] = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
        # for some architectures which requires a special fix for prompt tuning etc.
        # TODO: starting with transformers 4.38, all architectures should support caching.
        from mindone.transformers import __version__

        uses_transformers_4_38 = packaging.version.parse(__version__) >= packaging.version.parse("4.38.0")
        uses_transformers_4_36 = packaging.version.parse(__version__) >= packaging.version.parse("4.36.0")
        transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
        if packaging.version.parse(__version__) > packaging.version.parse("4.43.3"):
            # https://github.com/huggingface/transformers/pull/31445
            transformers_new_cache_archs.append("bloom")

        uses_cache = uses_transformers_4_38 or (
            uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
        )

        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        if peft_config.is_prompt_learning:
            if uses_cache and (model_kwargs.get("past_key_values", None) is not None):
                # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
                # In prompt learning methods, past key values are longer when compared to the `input_ids`.
                # As such only consider the last input ids in the autogressive generation phase.
                past_key_values = model_kwargs["past_key_values"]
                if isinstance(past_key_values, (tuple, list)):
                    seq_len = past_key_values[0][0].shape[-2]
                else:  # using transformers kv cache
                    seq_len = past_key_values.get_seq_length()
                if seq_len >= model_kwargs["input_ids"].shape[1]:
                    model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

            if model_kwargs.get("attention_mask", None) is not None:
                size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                prefix_attention_mask = mint.ones(size)
                model_kwargs["attention_mask"] = mint.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            # no past_key_values or past_key_values empty cache
            requires_prompt_injection = (model_kwargs.get("past_key_values", None) is None) or (
                isinstance(model_kwargs["past_key_values"], Cache)
                and not model_kwargs["past_key_values"].get_seq_length()
            )

            if requires_prompt_injection and peft_config.peft_type == PeftType.PREFIX_TUNING:
                new_past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = new_past_key_values
            elif requires_prompt_injection:
                inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                prompts = prompts.to(inputs_embeds.dtype)
                model_kwargs["inputs_embeds"] = mint.cat((prompts, inputs_embeds), dim=1)
                model_kwargs["input_ids"] = None

        # For transformers>=4.38.0 - for some architectures such as Llama, `cache_position` is
        # passed in the forward pass to keep track of the position ids of the cache. We have to
        # pop that from `model_kwargs` as `cache_position` is properly created by the model, using the passed
        # `inputs_embeds`: https://github.com/huggingface/transformers/blob/593230f0a1150ea9c0477b9d859f25daf73c8c33/src/transformers/models/llama/modeling_llama.py#L956  # noqa: E501
        _ = model_kwargs.pop("cache_position", None)

        return model_kwargs


@dataclass
class TunerLayerStatus:
    name: str
    module_type: str
    enabled: bool
    active_adapters: list[str]
    merged_adapters: list[str]
    requires_grad: dict[str, Union[bool, Literal["irregular"]]]
    available_adapters: list[str]


def get_layer_status(model: nn.Cell) -> list[TunerLayerStatus]:
    """Get the status of each adapter layer in the model.

    This function returns a list of `TunerLayerStatus` dataclass instances, each of which contains the following
    attributes:

    - `name` (`str`):
       The name of the adapter layer, e.g. `model.encoder.block.0.layer.0.SelfAttention.q`.
    - `module_type` (`str`):
       The type of the adapter layer, e.g. `lora.Linear`.
    - `enabled` (`bool`):
       Whether the adapter layer is enabled.
    - `active_adapters` (`list[str]`):
       The names of the active adapters, if any, e.g. `["default"]`.
    - `merged_adapters` (`list[str]`):
       The names of the merged adapters, if any, e.g. `["default"]`.
    - requires_grad : dict[str, Union[bool, Literal["irregular"]]]
       The requires_grad status of the parameters for each adapter module. Ideally, it should be either `True` or
       `False`. If the requires_grad status is not consistent across all parameters, the value will be set to
       `"irregular"`.
    - `available_adapters` (`list[str]`):
       The names of the available adapters, e.g. `["default"]`.
    - `devices` (`dict[str, list[str]]`):
       The devices where the parameters of the given adapter are stored, e.g. `["cuda"]`.

    Args:
        model ([Union[`~PeftModel`, `~transformers.PreTrainedModel`, `nn.Module`]]):
            The model to get the adapter layer status from.

    Returns:
        list[`peft.peft_model.TunerLayerStatus`]:
            A list of dataclasses, each containing the status of the corresponding adapter layer.

    """
    if isinstance(model, PeftModel):
        base_model = model.base_model
        if not isinstance(base_model, BaseTuner):
            raise TypeError(
                "get_layer_status() got an invalid PeftModel instance; prefix tuning and adaption prompt are not "
                "supported."
            )
    else:
        base_model = model

    layer_status: list[TunerLayerStatus] = []
    for name, module in base_model.name_cells():
        if not isinstance(module, BaseTunerLayer):
            continue

        # determine if all submodules/parameters if this module require grad or not
        mapping_requires_grad_list: dict[str, list[bool]] = collections.defaultdict(list)
        for adapter_module_name in module.adapter_layer_names:
            adapter_module = getattr(module, adapter_module_name)
            if isinstance(adapter_module, nn.CellDict):
                for key, submodule in adapter_module.items():
                    for param in submodule.get_parameters():
                        mapping_requires_grad_list[key].append(param.requires_grad)
            else:
                # strange, we don't know how to handle this, ignore for now
                pass

        def check_irrgular(vals: list[bool]) -> Union[bool, Literal["irregular"]]:
            if all(vals):
                return True
            if not any(vals):
                return False
            return "irregular"

        requires_grad = {key: check_irrgular(vals) for key, vals in mapping_requires_grad_list.items()}

        status = TunerLayerStatus(
            name=name,
            module_type=repr(module).partition("(")[0],
            enabled=not module.disable_adapters,
            active_adapters=module.active_adapters,
            merged_adapters=module.merged_adapters,
            requires_grad=requires_grad,
            available_adapters=sorted(module._get_available_adapters()),
        )
        layer_status.append(status)

    if not layer_status:
        raise ValueError(
            "No adapter layers found in the model, please ensure that it's a PEFT model or that you have PEFT adapters "
            "injected in the model."
        )

    return layer_status


@dataclass
class TunerModelStatus:
    base_model_type: str
    adapter_model_type: str
    peft_types: dict[str, str]
    trainable_params: int
    total_params: int
    num_adapter_layers: int
    enabled: Union[bool, Literal["irregular"]]
    active_adapters: Union[list[str], Literal["irregular"]]
    merged_adapters: Union[list[str], Literal["irregular"]]
    requires_grad: dict[str, Union[bool, Literal["irregular"]]]
    available_adapters: list[str]
    devices: dict[str, list[str]]


def get_model_status(model: nn.Cell) -> TunerModelStatus:
    """Get the status of tuners of the model.

    This function returns a `TunerModelStatus` dataclass instance, which contains the following attributes:

    - `base_model_type` (`str`):
       The type of the base model, e.g. `T5Model`.
    - `adapter_model_type` (`str`):
       The type of the adapter model, e.g. `LoraModel`.
    - `peft_types` (`dict[str, str]`):
       The mapping of adapter name to adapter type, e.g. `{"default": "LORA"}`.
    - `trainable_params` (`int`):
       The number of trainable parameters in the model.
    - `total_params` (`int`):
       The total number of parameters in the model.
    - `num_adapter_layers` (`int`):
       The number of adapter layers in the model.
    - `enabled` (`bool`, `Literal["irregular"]`):
       Whether all adapter layers are enabled. If some are enabled and some are not, this will be `"irregular"`. This
       means that your model is in an inconsistent state and might not work as expected.
    - `active_adapters` (`list[str]`, `Literal["irregular"]`):
       The names of the active adapters. If the active adapters are not consistent across all layers, this will be
       `"irregular"`, which means that your model is in an inconsistent state and might not work as expected.
    - `merged_adapters` (`list[str]`, `Literal["irregular"]`):
       The names of the merged adapters. If the merged adapters are not consistent across all layers, this will be
       `"irregular"`, which means that your model is in an inconsistent state and might not work as expected.
    - `requires_grad` (`dict[str, Union[bool, Literal["irregular"]]]`):
       Whether for the given adapter, all adapter layers have `requires_grad` set to `True` or `False`. If there is a
       mix, this will be set to `"irregular"`, which means that your model is in an inconsistent state and might not
       work as expected.
    - `available_adapters` (`list[str]`):
       The names of the available adapters, e.g. `["default"]`.
    - `devices` (`dict[str, list[str]]`):
       The devices where the parameters of the given adapter are stored, e.g. `["cuda"]`.

    Args:
        model ([Union[`~PeftModel`, `~transformers.PreTrainedModel`, `nn.Module`]]):
            The model to get the adapter layer status from.

    Returns:
        `peft.peft_model.TunerModelStatus`:
            A dataclass containing the status of the model.

    """
    if isinstance(model, PeftModel):
        if not isinstance(model.base_model, BaseTuner):
            raise TypeError(
                "get_model_status() got an invalid PeftModel instance; prefix tuning and adaption prompt are not "
                "supported."
            )
        base_model_type = model.get_base_model().__class__.__name__
        trainable_params, total_params = model.get_nb_trainable_parameters()
        base_model = model.base_model
        peft_types = {key: str(config.peft_type).partition(".")[-1] for key, config in base_model.peft_config.items()}
        adapter_model_type = base_model.__class__.__name__
    elif isinstance(model, PreTrainedModel):
        base_model_type = model.__class__.__name__
        trainable_params, total_params = PeftModel.get_nb_trainable_parameters(model)
        base_model = model
        peft_types = {}
        adapter_model_type = "None"
    else:
        base_model_type = "other"
        trainable_params, total_params = PeftModel.get_nb_trainable_parameters(model)
        base_model = model
        peft_types = {}
        adapter_model_type = "None"

    layer_status = get_layer_status(model)
    num_adapter_layers = len(layer_status)

    enabled_set: set[bool] = {status.enabled for status in layer_status}  # must be {True}, {False}, or {True, False}
    enabled: Union[bool, Literal["irregular"]]
    if len(enabled_set) == 1:
        enabled = enabled_set.pop()
    else:
        enabled = "irregular"

    available_adapters: list[str] = sorted(set().union(*(status.available_adapters for status in layer_status)))

    # ideally, active adapters should be consistent across all layers of the model, but we cannot guarantee it
    all_active_adapters: set[tuple[str, ...]] = {tuple(status.active_adapters) for status in layer_status}
    active_adapters: Union[list[str], Literal["irregular"]]
    if not all_active_adapters:
        active_adapters = []
    elif len(all_active_adapters) == 1:
        active_adapters = list(all_active_adapters.pop())
    else:
        active_adapters = "irregular"

    # Here we determine what adapters are merged. This is not trivial because multiple adapters can be merged or not at
    # the same time. Some layers may only have adapter A, some only adapter B, so it's not as easy as just checking
    # which adapters are merged on each layer.

    # First, determine all adapters that are merged on at least on module.
    merged_all: set[str] = set()
    for status in layer_status:
        merged_all.update(status.merged_adapters)

    # Next, check if on any layer, on of these adapters is not merged.
    merged_adapters: Union[list[str], Literal["irregular"]] = sorted(merged_all)
    for status in layer_status:
        unmerged = set(status.available_adapters) - set(status.merged_adapters)
        if unmerged & merged_all:
            # there is overlap between unmerged adapters and adapters that should be merged
            merged_adapters = "irregular"
            break

    # check status of requires_grad
    # first, merge the values for all layers
    requires_grad_all: dict[str, list[Union[bool, Literal["irregular"]]]] = collections.defaultdict(list)
    for status in layer_status:
        for key, val in status.requires_grad.items():
            requires_grad_all[key].append(val)

    # then, check if the values are consistent
    def check_irrgular(vals: list[Union[bool, Literal["irregular"]]]) -> Union[bool, Literal["irregular"]]:
        if all(val is True for val in vals):
            return True
        if all(val is False for val in vals):
            return False
        return "irregular"

    requires_grad = {key: check_irrgular(vals) for key, vals in requires_grad_all.items()}

    devices_dd = collections.defaultdict(list)
    for status in layer_status:
        for key, val in status.devices.items():
            devices_dd[key].extend(val)
    devices = {key: sorted(set(val)) for key, val in devices_dd.items()}

    adapter_model_status = TunerModelStatus(
        base_model_type=base_model_type,
        adapter_model_type=adapter_model_type,
        peft_types=peft_types,
        trainable_params=trainable_params,
        total_params=total_params,
        num_adapter_layers=num_adapter_layers,
        enabled=enabled,
        active_adapters=active_adapters,
        merged_adapters=merged_adapters,
        requires_grad=requires_grad,
        available_adapters=available_adapters,
        devices=devices,
    )
    return adapter_model_status


def __getattr__(name):
    if name == "PEFT_TYPE_TO_MODEL_MAPPING":
        # This is for backwards compatibility: In #2282, PEFT_TYPE_TO_MODEL_MAPPING was removed as it was redundant with
        # PEFT_TYPE_TO_TUNER_MAPPING. However, third party code could still use this mapping, e.g.:
        # https://github.com/AutoGPTQ/AutoGPTQ/blob/6689349625de973b9ee3016c28c11f32acf7f02c/auto_gptq/utils/peft_utils.py#L8
        # TODO: Remove after 2026-01
        msg = (
            "PEFT_TYPE_TO_MODEL_MAPPING is deprecated, please use `from peft import PEFT_TYPE_TO_TUNER_MAPPING` instead. "
            "The deprecated variable will be removed in 2026."
        )
        warnings.warn(msg, category=DeprecationWarning)
        return PEFT_TYPE_TO_TUNER_MAPPING

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
