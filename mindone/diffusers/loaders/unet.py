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

from ..utils import (
    _get_model_file,
    delete_adapter_layers,
    logging,
    set_adapter_layers,
    set_weights_and_activate_adapters,
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
