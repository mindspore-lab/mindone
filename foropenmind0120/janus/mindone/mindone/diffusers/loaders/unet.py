# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import os
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Union

from huggingface_hub.utils import validate_hf_hub_args

import mindspore as ms

from mindone.safetensors.mindspore import load_file, save_file

from ..models.embeddings import (
    ImageProjection,
    IPAdapterFaceIDImageProjection,
    IPAdapterFaceIDPlusImageProjection,
    IPAdapterFullImageProjection,
    IPAdapterPlusImageProjection,
    MultiIPAdapterImageProjection,
)
from ..utils import (
    _get_model_file,
    convert_unet_state_dict_to_peft,
    get_adapter_name,
    get_peft_kwargs,
    is_peft_version,
    logging,
)
from .lora_pipeline import LORA_WEIGHT_NAME, LORA_WEIGHT_NAME_SAFE, TEXT_ENCODER_NAME, UNET_NAME
from .single_file_utils import _load_param_into_net

logger = logging.get_logger(__name__)


CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"


class UNet2DConditionLoadersMixin:
    """
    Load LoRA layers into a [`UNet2DConditionModel`].
    """

    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    @validate_hf_hub_args
    def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, ms.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers into [`UNet2DConditionModel`]. Attention processor layers have to be
        defined in
        [`attention_processor.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
        and be a `mindspore.nn.Cell` class. Currently supported: LoRA, Custom Diffusion. For LoRA, one must install
        `peft`: `pip install -U peft`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the model id (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a directory (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A mindspore state dict.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            network_alphas (`Dict[str, float]`):
                The value of the network alpha used for stable learning and preventing underflow. This value has the
                same meaning as the `--network_alpha` option in the kohya-ss trainer script. Refer to [this
                link](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning).
            adapter_name (`str`, *optional*, defaults to None):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            weight_name (`str`, *optional*, defaults to None):
                Name of the serialized state dict file.

        Example:

        ```py
        from mindone.diffusers import AutoPipelineForText2Image
        import mindspore as ms

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16
        )
        pipeline.unet.load_attn_procs(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        adapter_name = kwargs.pop("adapter_name", None)
        _pipeline = kwargs.pop("_pipeline", None)
        network_alphas = kwargs.pop("network_alphas", None)
        allow_pickle = False

        _pipeline = kwargs.pop("_pipeline", None)  # noqa: F841

        is_network_alphas_none = network_alphas is None  # noqa: F841

        allow_pickle = False

        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = load_file(model_file)
                except IOError as e:
                    if not allow_pickle:
                        raise e
                    # try loading non-safetensors weights
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                raise NotImplementedError(
                    f"Only supports deserialization of weights file in safetensors format, but got {model_file}"
                )
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())
        is_lora = all(("lora" in k or k.endswith(".alpha")) for k in state_dict.keys())

        if is_custom_diffusion:
            raise NotImplementedError("CustomDiffusionAttnProcessor is not yet supported.")
        elif is_lora:
            is_model_cpu_offload, is_sequential_cpu_offload = self._process_lora(
                state_dict=state_dict,
                unet_identifier_key=self.unet_name,
                network_alphas=network_alphas,
                adapter_name=adapter_name,
                _pipeline=_pipeline,
            )
        else:
            raise ValueError(
                f"{model_file} does not seem to be in the correct format expected by Custom Diffusion training."
            )

    def _process_lora(self, state_dict, unet_identifier_key, network_alphas, adapter_name, _pipeline):
        # This method does the following things:
        # 1. Filters the `state_dict` with keys matching  `unet_identifier_key` when using the non-legacy
        #    format. For legacy format no filtering is applied.
        # 2. Converts the `state_dict` to the `peft` compatible format.
        # 3. Creates a `LoraConfig` and then injects the converted `state_dict` into the UNet per the
        #    `LoraConfig` specs.
        # 4. It also reports if the underlying `_pipeline` has any kind of offloading inside of it.
        from mindone.diffusers._peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

        keys = list(state_dict.keys())

        unet_keys = [k for k in keys if k.startswith(unet_identifier_key)]
        unet_state_dict = {k.replace(f"{unet_identifier_key}.", ""): v for k, v in state_dict.items() if k in unet_keys}

        if network_alphas is not None:
            alpha_keys = [k for k in network_alphas.keys() if k.startswith(unet_identifier_key)]
            network_alphas = {
                k.replace(f"{unet_identifier_key}.", ""): v for k, v in network_alphas.items() if k in alpha_keys
            }

        is_model_cpu_offload = False
        is_sequential_cpu_offload = False
        state_dict_to_be_used = unet_state_dict if len(unet_state_dict) > 0 else state_dict

        if len(state_dict_to_be_used) > 0:
            if adapter_name in getattr(self, "peft_config", {}):
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the Unet - please select a new adapter name."
                )

            state_dict = convert_unet_state_dict_to_peft(state_dict_to_be_used)

            if network_alphas is not None:
                # The alphas state dict have the same structure as Unet, thus we convert it to peft format using
                # `convert_unet_state_dict_to_peft` method.
                network_alphas = convert_unet_state_dict_to_peft(network_alphas)

            rank = {}
            for key, val in state_dict.items():
                if "lora_B" in key:
                    rank[key] = val.shape[1]

            lora_config_kwargs = get_peft_kwargs(rank, network_alphas, state_dict, is_unet=True)
            if "use_dora" in lora_config_kwargs:
                if lora_config_kwargs["use_dora"]:
                    if is_peft_version("<", "0.9.0"):
                        raise ValueError(
                            "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                        )
                else:
                    if is_peft_version("<", "0.9.0"):
                        lora_config_kwargs.pop("use_dora")
            lora_config = LoraConfig(**lora_config_kwargs)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(self)

            inject_adapter_in_model(lora_config, self, adapter_name=adapter_name)
            incompatible_keys = set_peft_model_state_dict(self, state_dict, adapter_name)

            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        return is_model_cpu_offload, is_sequential_cpu_offload

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        **kwargs,
    ):
        r"""
        Save attention processor layers to a directory so that it can be reloaded with the
        [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save an attention processor to (will be created if it doesn't exist).
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `MindSpore.save_checkpoint` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or with `pickle`.

        Example:

        ```py
        import mindspore
        from mindone.diffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            mindspore_dtype=mindspore.float16,
        )
        pipeline.unet.load_attn_procs("path-to-save-model", weight_name="lora_diffusion_weights.safetensors")
        pipeline.unet.save_attn_procs("path-to-save-model", weight_name="lora_diffusion_weights.safetensors")
        ```
        """
        from ..models.attention_processor import CustomDiffusionAttnProcessor

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        is_custom_diffusion = any(
            isinstance(x, CustomDiffusionAttnProcessor) for (_, x) in self.attn_processors.items()
        )
        if is_custom_diffusion:
            raise NotImplementedError(
                f"is_custom_diffusion is not yet supported in {self.__class__.__name__}.save_attn_procs ."
            )
        else:
            from mindone.diffusers._peft.utils import get_peft_model_state_dict

            state_dict = get_peft_model_state_dict(self)

        if save_function is None:
            if safe_serialization:
                save_function = partial(save_file, metadata={"format": "np"})
            else:
                save_function = ms.save_checkpoint

        os.makedirs(save_directory, exist_ok=True)

        if weight_name is None:
            if safe_serialization:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE if is_custom_diffusion else LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME if is_custom_diffusion else LORA_WEIGHT_NAME

        # Save the model
        save_path = Path(save_directory, weight_name).as_posix()
        save_function(state_dict, save_path)
        logger.info(f"Model weights saved in {save_path}")

    def _convert_ip_adapter_image_proj_to_diffusers(self, state_dict):
        updated_state_dict = {}
        image_projection = None

        if "proj.weight" in state_dict:
            # IP-Adapter
            num_image_text_embeds = 4
            clip_embeddings_dim = state_dict["proj.weight"].shape[-1]
            cross_attention_dim = state_dict["proj.weight"].shape[0] // 4

            image_projection = ImageProjection(
                cross_attention_dim=cross_attention_dim,
                image_embed_dim=clip_embeddings_dim,
                num_image_text_embeds=num_image_text_embeds,
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj", "image_embeds")
                updated_state_dict[diffusers_name] = value

        elif "proj.3.weight" in state_dict:
            # IP-Adapter Full
            clip_embeddings_dim = state_dict["proj.0.weight"].shape[0]
            cross_attention_dim = state_dict["proj.3.weight"].shape[0]

            image_projection = IPAdapterFullImageProjection(
                cross_attention_dim=cross_attention_dim, image_embed_dim=clip_embeddings_dim
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj.0", "ff.net.0.proj")
                diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
                diffusers_name = diffusers_name.replace("proj.3", "norm")
                updated_state_dict[diffusers_name] = value
        elif "perceiver_resampler.proj_in.weight" in state_dict:
            # IP-Adapter Face ID Plus
            id_embeddings_dim = state_dict["proj.0.weight"].shape[1]
            embed_dims = state_dict["perceiver_resampler.proj_in.weight"].shape[0]
            hidden_dims = state_dict["perceiver_resampler.proj_in.weight"].shape[1]
            output_dims = state_dict["perceiver_resampler.proj_out.weight"].shape[0]
            heads = state_dict["perceiver_resampler.layers.0.0.to_q.weight"].shape[0] // 64

            image_projection = IPAdapterFaceIDPlusImageProjection(
                embed_dims=embed_dims,
                output_dims=output_dims,
                hidden_dims=hidden_dims,
                heads=heads,
                id_embeddings_dim=id_embeddings_dim,
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("perceiver_resampler.", "")
                diffusers_name = diffusers_name.replace("0.to", "attn.to")
                diffusers_name = diffusers_name.replace("0.1.0.", "0.ff.0.")
                diffusers_name = diffusers_name.replace("0.1.1.weight", "0.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("0.1.3.weight", "0.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("1.1.0.", "1.ff.0.")
                diffusers_name = diffusers_name.replace("1.1.1.weight", "1.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("1.1.3.weight", "1.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("2.1.0.", "2.ff.0.")
                diffusers_name = diffusers_name.replace("2.1.1.weight", "2.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("2.1.3.weight", "2.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("3.1.0.", "3.ff.0.")
                diffusers_name = diffusers_name.replace("3.1.1.weight", "3.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("3.1.3.weight", "3.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("layers.0.0", "layers.0.ln0")
                diffusers_name = diffusers_name.replace("layers.0.1", "layers.0.ln1")
                diffusers_name = diffusers_name.replace("layers.1.0", "layers.1.ln0")
                diffusers_name = diffusers_name.replace("layers.1.1", "layers.1.ln1")
                diffusers_name = diffusers_name.replace("layers.2.0", "layers.2.ln0")
                diffusers_name = diffusers_name.replace("layers.2.1", "layers.2.ln1")
                diffusers_name = diffusers_name.replace("layers.3.0", "layers.3.ln0")
                diffusers_name = diffusers_name.replace("layers.3.1", "layers.3.ln1")

                if "norm1" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm1", "0")] = value
                elif "norm2" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm2", "1")] = value
                elif "to_kv" in diffusers_name:
                    v_chunk = value.chunk(2, axis=0)
                    updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = ms.Parameter(
                        v_chunk[0], name=diffusers_name.replace("to_kv", "to_k")
                    )
                    updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = ms.Parameter(
                        v_chunk[1], name=diffusers_name.replace("to_kv", "to_v")
                    )
                elif "to_out" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
                elif "proj.0.weight" == diffusers_name:
                    updated_state_dict["proj.net.0.proj.weight"] = value
                elif "proj.0.bias" == diffusers_name:
                    updated_state_dict["proj.net.0.proj.bias"] = value
                elif "proj.2.weight" == diffusers_name:
                    updated_state_dict["proj.net.2.weight"] = value
                elif "proj.2.bias" == diffusers_name:
                    updated_state_dict["proj.net.2.bias"] = value
                else:
                    updated_state_dict[diffusers_name] = value

        elif "norm.weight" in state_dict:
            # IP-Adapter Face ID
            id_embeddings_dim_in = state_dict["proj.0.weight"].shape[1]
            id_embeddings_dim_out = state_dict["proj.0.weight"].shape[0]
            multiplier = id_embeddings_dim_out // id_embeddings_dim_in
            norm_layer = "norm.weight"
            cross_attention_dim = state_dict[norm_layer].shape[0]
            num_tokens = state_dict["proj.2.weight"].shape[0] // cross_attention_dim

            image_projection = IPAdapterFaceIDImageProjection(
                cross_attention_dim=cross_attention_dim,
                image_embed_dim=id_embeddings_dim_in,
                mult=multiplier,
                num_tokens=num_tokens,
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj.0", "ff.net.0.proj")
                diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
                updated_state_dict[diffusers_name] = value

        else:
            # IP-Adapter Plus
            num_image_text_embeds = state_dict["latents"].shape[1]
            embed_dims = state_dict["proj_in.weight"].shape[1]
            output_dims = state_dict["proj_out.weight"].shape[0]
            hidden_dims = state_dict["latents"].shape[2]
            attn_key_present = any("attn" in k for k in state_dict)
            heads = (
                state_dict["layers.0.attn.to_q.weight"].shape[0] // 64
                if attn_key_present
                else state_dict["layers.0.0.to_q.weight"].shape[0] // 64
            )

            image_projection = IPAdapterPlusImageProjection(
                embed_dims=embed_dims,
                output_dims=output_dims,
                hidden_dims=hidden_dims,
                heads=heads,
                num_queries=num_image_text_embeds,
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("0.to", "2.to")

                diffusers_name = diffusers_name.replace("0.0.norm1", "0.ln0")
                diffusers_name = diffusers_name.replace("0.0.norm2", "0.ln1")
                diffusers_name = diffusers_name.replace("1.0.norm1", "1.ln0")
                diffusers_name = diffusers_name.replace("1.0.norm2", "1.ln1")
                diffusers_name = diffusers_name.replace("2.0.norm1", "2.ln0")
                diffusers_name = diffusers_name.replace("2.0.norm2", "2.ln1")
                diffusers_name = diffusers_name.replace("3.0.norm1", "3.ln0")
                diffusers_name = diffusers_name.replace("3.0.norm2", "3.ln1")

                if "to_kv" in diffusers_name:
                    parts = diffusers_name.split(".")
                    parts[2] = "attn"
                    diffusers_name = ".".join(parts)
                    v_chunk = value.chunk(2, axis=0)
                    updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = ms.Parameter(
                        v_chunk[0], name=diffusers_name.replace("to_kv", "to_k")
                    )
                    updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = ms.Parameter(
                        v_chunk[1], name=diffusers_name.replace("to_kv", "to_v")
                    )
                elif "to_q" in diffusers_name:
                    parts = diffusers_name.split(".")
                    parts[2] = "attn"
                    diffusers_name = ".".join(parts)
                    updated_state_dict[diffusers_name] = value
                elif "to_out" in diffusers_name:
                    parts = diffusers_name.split(".")
                    parts[2] = "attn"
                    diffusers_name = ".".join(parts)
                    updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
                else:
                    diffusers_name = diffusers_name.replace("0.1.0", "0.ff.0")
                    diffusers_name = diffusers_name.replace("0.1.1", "0.ff.1.net.0.proj")
                    diffusers_name = diffusers_name.replace("0.1.3", "0.ff.1.net.2")

                    diffusers_name = diffusers_name.replace("1.1.0", "1.ff.0")
                    diffusers_name = diffusers_name.replace("1.1.1", "1.ff.1.net.0.proj")
                    diffusers_name = diffusers_name.replace("1.1.3", "1.ff.1.net.2")

                    diffusers_name = diffusers_name.replace("2.1.0", "2.ff.0")
                    diffusers_name = diffusers_name.replace("2.1.1", "2.ff.1.net.0.proj")
                    diffusers_name = diffusers_name.replace("2.1.3", "2.ff.1.net.2")

                    diffusers_name = diffusers_name.replace("3.1.0", "3.ff.0")
                    diffusers_name = diffusers_name.replace("3.1.1", "3.ff.1.net.0.proj")
                    diffusers_name = diffusers_name.replace("3.1.3", "3.ff.1.net.2")
                    updated_state_dict[diffusers_name] = value
                    updated_state_dict[diffusers_name] = value
        _load_param_into_net(image_projection, updated_state_dict)

        return image_projection

    def _convert_ip_adapter_attn_to_diffusers(self, state_dicts):
        from ..models.attention_processor import IPAdapterAttnProcessor

        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        key_id = 1

        for name in self.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.config.block_out_channels[block_id]

            if cross_attention_dim is None or "motion_modules" in name:
                attn_processor_class = self.attn_processors[name].__class__
                attn_procs[name] = attn_processor_class()

            else:
                attn_processor_class = IPAdapterAttnProcessor
                num_image_text_embeds = []
                for state_dict in state_dicts:
                    if "proj.weight" in state_dict["image_proj"]:
                        # IP-Adapter
                        num_image_text_embeds += [4]
                    elif "proj.3.weight" in state_dict["image_proj"]:
                        # IP-Adapter Full Face
                        num_image_text_embeds += [257]  # 256 CLIP tokens + 1 CLS token
                    elif "perceiver_resampler.proj_in.weight" in state_dict["image_proj"]:
                        # IP-Adapter Face ID Plus
                        num_image_text_embeds += [4]
                    elif "norm.weight" in state_dict["image_proj"]:
                        # IP-Adapter Face ID
                        num_image_text_embeds += [4]
                    else:
                        # IP-Adapter Plus
                        num_image_text_embeds += [state_dict["image_proj"]["latents"].shape[1]]

                attn_procs[name] = attn_processor_class(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=num_image_text_embeds,
                )
                value_dict = {}
                for i, state_dict in enumerate(state_dicts):
                    value_dict.update({f"to_k_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
                    value_dict.update({f"to_v_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})

                _load_param_into_net(attn_procs[name], value_dict)

                key_id += 2

        return attn_procs

    def _load_ip_adapter_weights(self, state_dicts):
        if not isinstance(state_dicts, list):
            state_dicts = [state_dicts]

        # Kolors Unet already has a `encoder_hid_proj`
        if (
            self.encoder_hid_proj is not None
            and self.config.encoder_hid_dim_type == "text_proj"
            and not hasattr(self, "text_encoder_hid_proj")
        ):
            self.has_text_encoder_hid_proj = True
            self.text_encoder_hid_proj = self.encoder_hid_proj

        # Set encoder_hid_proj after loading ip_adapter weights,
        # because `IPAdapterPlusImageProjection` also has `attn_processors`.
        self.encoder_hid_proj = None

        attn_procs = self._convert_ip_adapter_attn_to_diffusers(state_dicts)
        self.set_attn_processor(attn_procs)

        # convert IP-Adapter Image Projection layers to diffusers
        image_projection_layers = []
        for state_dict in state_dicts:
            image_projection_layer = self._convert_ip_adapter_image_proj_to_diffusers(state_dict["image_proj"])
            image_projection_layers.append(image_projection_layer)

        self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        self.config.encoder_hid_dim_type = "ip_image_proj"
        self.config["encoder_hid_dim_type"] = "ip_image_proj"  # not same with `self.config.encoder_hid_dim_type`
        self.encoder_hid_dim_type = "ip_image_proj"  # used in UNet2DConditionModel.construct()
        self.to(dtype=self.dtype)

    def _load_ip_adapter_loras(self, state_dicts):
        lora_dicts = {}
        for key_id, name in enumerate(self.attn_processors.keys()):
            for i, state_dict in enumerate(state_dicts):
                if f"{key_id}.to_k_lora.down.weight" in state_dict["ip_adapter"]:
                    if i not in lora_dicts:
                        lora_dicts[i] = {}
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_k_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_k_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_q_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_q_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_v_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_v_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_out_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_out_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_k_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_k_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_q_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_q_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_v_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_v_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_out_lora.up.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_out_lora.up.weight"
                            ]
                        }
                    )
        return lora_dicts
