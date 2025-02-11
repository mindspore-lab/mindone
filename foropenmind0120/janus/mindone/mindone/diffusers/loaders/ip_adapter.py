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

from pathlib import Path
from typing import Dict, List, Optional, Union

from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPImageProcessor

import mindspore

from mindone.safetensors.mindspore import load_file
from mindone.transformers import CLIPVisionModelWithProjection

from ..models.attention_processor import AttnProcessor, IPAdapterAttnProcessor
from ..utils import _get_model_file, logging
from .unet_loader_utils import _maybe_expand_lora_scales

logger = logging.get_logger(__name__)


class IPAdapterMixin:
    """Mixin for handling IP Adapters."""

    @validate_hf_hub_args
    def load_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, List[str], Dict[str, mindspore.Tensor]],
        subfolder: Union[str, List[str]],
        weight_name: Union[str, List[str]],
        image_encoder_folder: Optional[str] = "image_encoder",
        **kwargs,
    ):
        """
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `List[str]` or `os.PathLike` or `List[os.PathLike]` or `dict` or `List[dict]`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [mindspore state dict]
            subfolder (`str` or `List[str]`):
                The subfolder location of a model file within a larger model repository on the Hub or locally. If a
                list is passed, it should have the same length as `weight_name`.
            weight_name (`str` or `List[str]`):
                The name of the weight file to load. If a list is passed, it should have the same length as
                `weight_name`.
            image_encoder_folder (`str`, *optional*, defaults to `image_encoder`):
                The subfolder location of the image encoder within a larger model repository on the Hub or locally.
                Pass `None` to not load the image encoder. If the image encoder is located in a folder inside
                `subfolder`, you only need to pass the name of the folder that contains image encoder weights, e.g.
                `image_encoder_folder="image_encoder"`. If the image encoder is located in a folder other than
                `subfolder`, you should pass the path to the folder that contains image encoder weights, for example,
                `image_encoder_folder="different_subfolder/image_encoder"`.
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
        """

        # handle the list inputs for multiple IP Adapters
        if not isinstance(weight_name, list):
            weight_name = [weight_name]

        if not isinstance(pretrained_model_name_or_path_or_dict, list):
            pretrained_model_name_or_path_or_dict = [pretrained_model_name_or_path_or_dict]
        if len(pretrained_model_name_or_path_or_dict) == 1:
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict * len(weight_name)

        if not isinstance(subfolder, list):
            subfolder = [subfolder]
        if len(subfolder) == 1:
            subfolder = subfolder * len(weight_name)

        if len(weight_name) != len(pretrained_model_name_or_path_or_dict):
            raise ValueError("`weight_name` and `pretrained_model_name_or_path_or_dict` must have the same length.")

        if len(weight_name) != len(subfolder):
            raise ValueError("`weight_name` and `subfolder` must have the same length.")

        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        state_dicts = []
        for pretrained_model_name_or_path_or_dict, weight_name, subfolder in zip(
            pretrained_model_name_or_path_or_dict, weight_name, subfolder
        ):
            if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                )
                if weight_name.endswith(".safetensors"):
                    state_dict = {"image_proj": {}, "ip_adapter": {}}
                    for key, value in load_file(model_file).items():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = value
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = value
                else:
                    raise NotImplementedError(
                        f"Only supports deserialization of weights file in safetensors format, but got {model_file}"
                    )
            else:
                state_dict = pretrained_model_name_or_path_or_dict

            keys = list(state_dict.keys())
            if keys != ["image_proj", "ip_adapter"]:
                raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")

            state_dicts.append(state_dict)

            # load CLIP image encoder here if it has not been registered to the pipeline yet
            if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is None:
                if image_encoder_folder is not None:
                    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                        logger.info(f"loading image_encoder from {pretrained_model_name_or_path_or_dict}")
                        if image_encoder_folder.count("/") == 0:
                            image_encoder_subfolder = Path(subfolder, image_encoder_folder).as_posix()
                        else:
                            image_encoder_subfolder = Path(image_encoder_folder).as_posix()

                        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                            pretrained_model_name_or_path_or_dict,
                            subfolder=image_encoder_subfolder,
                            cache_dir=cache_dir,
                            local_files_only=local_files_only,
                        ).to(self.dtype)
                        self.register_modules(image_encoder=image_encoder)
                    else:
                        raise ValueError(
                            "`image_encoder` cannot be loaded because `pretrained_model_name_or_path_or_dict` is a state dict."
                        )
                else:
                    logger.warning(
                        "image_encoder is not loaded since `image_encoder_folder=None` passed."
                        "You will not be able to use `ip_adapter_image` when calling the pipeline with IP-Adapter."
                        "Use `ip_adapter_image_embeds` to pass pre-generated image embedding instead."
                    )

            # create feature extractor if it has not been registered to the pipeline yet
            if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is None:
                clip_image_size = self.image_encoder.config.image_size
                feature_extractor = CLIPImageProcessor(size=clip_image_size, crop_size=clip_image_size)
                self.register_modules(feature_extractor=feature_extractor)

        # load ip-adapter into unet
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        unet._load_ip_adapter_weights(state_dicts)

        extra_loras = unet._load_ip_adapter_loras(state_dicts)
        if extra_loras != {}:
            # apply the IP Adapter Face ID LoRA weights
            peft_config = getattr(unet, "peft_config", {})
            for k, lora in extra_loras.items():
                if f"faceid_{k}" not in peft_config:
                    self.load_lora_weights(lora, adapter_name=f"faceid_{k}")
                    self.set_adapters([f"faceid_{k}"], adapter_weights=[1.0])

    def set_ip_adapter_scale(self, scale):
        """
        Set IP-Adapter scales per-transformer block. Input `scale` could be a single config or a list of configs for
        granular control over each IP-Adapter behavior. A config can be a float or a dictionary.

        Example:

        ```py
        # To use original IP-Adapter
        scale = 1.0
        pipeline.set_ip_adapter_scale(scale)

        # To use style block only
        scale = {
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        pipeline.set_ip_adapter_scale(scale)

        # To use style+layout blocks
        scale = {
            "down": {"block_2": [0.0, 1.0]},
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        pipeline.set_ip_adapter_scale(scale)

        # To use style and layout from 2 reference images
        scales = [{"down": {"block_2": [0.0, 1.0]}}, {"up": {"block_0": [0.0, 1.0, 0.0]}}]
        pipeline.set_ip_adapter_scale(scales)
        ```
        """
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        if not isinstance(scale, list):
            scale = [scale]
        scale_configs = _maybe_expand_lora_scales(unet, scale, default_scale=0.0)

        for attn_name, attn_processor in unet.attn_processors.items():
            if isinstance(attn_processor, (IPAdapterAttnProcessor)):
                if len(scale_configs) != len(attn_processor.scale):
                    raise ValueError(
                        f"Cannot assign {len(scale_configs)} scale_configs to "
                        f"{len(attn_processor.scale)} IP-Adapter."
                    )
                elif len(scale_configs) == 1:
                    scale_configs = scale_configs * len(attn_processor.scale)
                for i, scale_config in enumerate(scale_configs):
                    if isinstance(scale_config, dict):
                        for k, s in scale_config.items():
                            if attn_name.startswith(k):
                                attn_processor.scale[i] = s
                    else:
                        attn_processor.scale[i] = scale_config

    def unload_ip_adapter(self):
        """
        Unloads the IP Adapter weights

        Examples:

        ```python
        >>> # Assuming `pipeline` is already loaded with the IP Adapter weights.
        >>> pipeline.unload_ip_adapter()
        >>> ...
        ```
        """
        # remove CLIP image encoder
        if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is not None:
            self.image_encoder = None
            self.register_to_config(image_encoder=[None, None])

        # remove feature extractor only when safety_checker is None as safety_checker uses
        # the feature_extractor later
        if not hasattr(self, "safety_checker"):
            if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is not None:
                self.feature_extractor = None
                self.register_to_config(feature_extractor=[None, None])

        # remove hidden encoder
        self.unet.encoder_hid_proj = None
        self.unet.config.encoder_hid_dim_type = None

        # Kolors: restore `encoder_hid_proj` with `text_encoder_hid_proj`
        if hasattr(self.unet, "text_encoder_hid_proj") and self.unet.text_encoder_hid_proj is not None:
            self.unet.encoder_hid_proj = self.unet.text_encoder_hid_proj
            self.unet.text_encoder_hid_proj = None
            self.unet.config.encoder_hid_dim_type = "text_proj"

        # restore original Unet attention processors layers
        attn_procs = {}
        for name, value in self.unet.attn_processors.items():
            attn_processor_class = AttnProcessor()
            attn_procs[name] = attn_processor_class if isinstance(value, IPAdapterAttnProcessor) else value.__class__()
        self.unet.set_attn_processor(attn_procs)
