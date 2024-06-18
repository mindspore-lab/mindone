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
import inspect
import os
from functools import partial
from typing import Callable, Dict, List, Optional, Union

from huggingface_hub.utils import validate_hf_hub_args

import mindspore as ms

from mindone.safetensors.mindspore import load_file

from ..models.embeddings import (
    ImageProjection,
    IPAdapterFullImageProjection,
    IPAdapterPlusImageProjection,
    MultiIPAdapterImageProjection,
)
from ..utils import (
    _get_model_file,
    delete_adapter_layers,
    logging,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)
from .single_file_utils import (
    _load_param_into_net,
    convert_stable_cascade_unet_single_file_to_diffusers,
    infer_stable_cascade_single_file_config,
    load_single_file_model_checkpoint,
)

logger = logging.get_logger(__name__)


TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

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
        and be a `torch.nn.Module` class.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the model id (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a directory (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
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
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.unet.load_attn_procs(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        network_alphas = kwargs.pop("network_alphas", None)

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
                        resume_download=resume_download,
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
                    resume_download=resume_download,
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
        if is_custom_diffusion:
            raise NotImplementedError("CustomDiffusionAttnProcessor is not yet supported.")
        # In fact, we have nothing to do as loading the adapter weights is already handled above
        # by `set_peft_model_state_dict` on the Unet

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        **kwargs,
    ):
        raise NotImplementedError(f"{self.__class__.__name__}.save_attn_procs is not yet supported.")

    def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
        self.lora_scale = lora_scale
        self._safe_fusing = safe_fusing
        self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))

    def _fuse_lora_apply(self, module, adapter_names=None):
        from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer

        merge_kwargs = {"safe_merge": self._safe_fusing}

        if isinstance(module, BaseTunerLayer):
            if self.lora_scale != 1.0:
                module.scale_layer(self.lora_scale)

            # For BC with previous PEFT versions, we need to check the signature
            # of the `merge` method to see if it supports the `adapter_names` argument.
            supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
            if "adapter_names" in supported_merge_kwargs:
                merge_kwargs["adapter_names"] = adapter_names
            elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                raise RuntimeError("The `adapter_names` argument is not supported with `BaseTunerLayer.merge`")

            module.merge(**merge_kwargs)

    def unfuse_lora(self):
        self.apply(self._unfuse_lora_apply)

    def _unfuse_lora_apply(self, module):
        from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer

        if isinstance(module, BaseTunerLayer):
            module.unmerge()

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[List[float], float]] = None,
    ):
        """
        Set the currently active adapters for use in the UNet.

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            weights (`Union[List[float], float]`, *optional*):
                The adapter(s) weights to use with the UNet. If `None`, the weights are set to `1.0` for all the
                adapters.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])
        ```
        """
        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        if weights is None:
            weights = [1.0] * len(adapter_names)
        elif isinstance(weights, float):
            weights = [weights] * len(adapter_names)

        if len(adapter_names) != len(weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
            )

        set_weights_and_activate_adapters(self, adapter_names, weights)

    def disable_lora(self):
        """
        Disable the UNet's active LoRA layers.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.disable_lora()
        ```
        """
        set_adapter_layers(self, enabled=False)

    def enable_lora(self):
        """
        Enable the UNet's active LoRA layers.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.enable_lora()
        ```
        """
        set_adapter_layers(self, enabled=True)

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Delete an adapter's LoRA layers from the UNet.

        Args:
            adapter_names (`Union[List[str], str]`):
                The names (single string or list of strings) of the adapter to delete.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_names="cinematic"
        )
        pipeline.delete_adapters("cinematic")
        ```
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for adapter_name in adapter_names:
            delete_adapter_layers(self, adapter_name)

            # Pop also the corresponding adapter from the config
            if hasattr(self, "peft_config"):
                self.peft_config.pop(adapter_name, None)

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

        else:
            # IP-Adapter Plus
            num_image_text_embeds = state_dict["latents"].shape[1]
            embed_dims = state_dict["proj_in.weight"].shape[1]
            output_dims = state_dict["proj_out.weight"].shape[0]
            hidden_dims = state_dict["latents"].shape[2]
            heads = state_dict["layers.0.0.to_q.weight"].shape[0] // 64

            image_projection = IPAdapterPlusImageProjection(
                embed_dims=embed_dims,
                output_dims=output_dims,
                hidden_dims=hidden_dims,
                heads=heads,
                num_queries=num_image_text_embeds,
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("0.to", "2.to")
                diffusers_name = diffusers_name.replace("1.0.weight", "3.0.weight")
                diffusers_name = diffusers_name.replace("1.0.bias", "3.0.bias")
                diffusers_name = diffusers_name.replace("1.1.weight", "3.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("1.3.weight", "3.1.net.2.weight")

                if "norm1" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm1", "0")] = value
                elif "norm2" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm2", "1")] = value
                elif "to_kv" in diffusers_name:
                    v_chunk = value.chunk(2, dim=0)
                    updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                    updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
                elif "to_out" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
                else:
                    updated_state_dict[diffusers_name] = value
        _load_param_into_net(image_projection, updated_state_dict)

        return image_projection

    def _convert_ip_adapter_attn_to_diffusers(self, state_dicts):
        from ..models.attention_processor import AttnProcessor, IPAdapterAttnProcessor

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
                attn_processor_class = AttnProcessor
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
        self.encoder_hid_dim_type = "ip_image_proj"
        self.to(dtype=self.dtype)


class FromOriginalUNetMixin:
    """
    Load pretrained UNet model weights saved in the `.ckpt` or `.safetensors` format into a [`StableCascadeUNet`].
    """

    @classmethod
    @validate_hf_hub_args
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`StableCascadeUNet`] from pretrained StableCascadeUNet weights saved in the original `.ckpt` or
        `.safetensors` format. The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            config: (`dict`, *optional*):
                Dictionary containing the configuration of the model:
            mindspore_dtype (`str` or `mindspore.dtype`, *optional*):
                Override the default `mindspore.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to True, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables of the model.

        """
        class_name = cls.__name__
        if class_name != "StableCascadeUNet":
            raise ValueError("FromOriginalUNetMixin is currently only compatible with StableCascadeUNet")

        config = kwargs.pop("config", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", None)
        revision = kwargs.pop("revision", None)
        mindspore_dtype = kwargs.pop("mindspore_dtype", None)

        checkpoint = load_single_file_model_checkpoint(
            pretrained_model_link_or_path,
            resume_download=resume_download,
            force_download=force_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )

        if config is None:
            config = infer_stable_cascade_single_file_config(checkpoint)
            model_config = cls.load_config(**config, **kwargs)
        else:
            model_config = config

        model = cls.from_config(model_config, **kwargs)

        diffusers_format_checkpoint = convert_stable_cascade_unet_single_file_to_diffusers(checkpoint)

        _load_param_into_net(model, diffusers_format_checkpoint)

        if mindspore_dtype is not None:
            model.to(mindspore_dtype)

        return model
