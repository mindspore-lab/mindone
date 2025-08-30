# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from ..models.embeddings import ImageProjection, MultiIPAdapterImageProjection
from ..models.model_loading_utils import _load_state_dict_into_model
from ..utils import logging

logger = logging.get_logger(__name__)


class FluxTransformer2DLoadersMixin:
    """
    Load layers into a [`FluxTransformer2DModel`].
    """

    def _convert_ip_adapter_image_proj_to_diffusers(self, state_dict, low_cpu_mem_usage=False):
        updated_state_dict = {}
        image_projection = None

        if "proj.weight" in state_dict:
            # IP-Adapter
            num_image_text_embeds = 4
            if state_dict["proj.weight"].shape[0] == 65536:
                num_image_text_embeds = 16
            clip_embeddings_dim = state_dict["proj.weight"].shape[-1]
            cross_attention_dim = state_dict["proj.weight"].shape[0] // num_image_text_embeds

            image_projection = ImageProjection(
                cross_attention_dim=cross_attention_dim,
                image_embed_dim=clip_embeddings_dim,
                num_image_text_embeds=num_image_text_embeds,
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj", "image_embeds")
                updated_state_dict[diffusers_name] = value

        _load_state_dict_into_model(image_projection, updated_state_dict)

        return image_projection

    def _convert_ip_adapter_attn_to_diffusers(self, state_dicts, low_cpu_mem_usage=False):
        from ..models.transformers.transformer_flux import FluxIPAdapterAttnProcessor

        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        key_id = 0
        for name in self.attn_processors.keys():
            if name.startswith("single_transformer_blocks"):
                attn_processor_class = self.attn_processors[name].__class__
                attn_procs[name] = attn_processor_class()
            else:
                cross_attention_dim = self.config.joint_attention_dim
                hidden_size = self.inner_dim
                attn_processor_class = FluxIPAdapterAttnProcessor
                num_image_text_embeds = []
                for state_dict in state_dicts:
                    if "proj.weight" in state_dict["image_proj"]:
                        num_image_text_embed = 4
                        if state_dict["image_proj"]["proj.weight"].shape[0] == 65536:
                            num_image_text_embed = 16
                        # IP-Adapter
                        num_image_text_embeds += [num_image_text_embed]

                attn_procs[name] = attn_processor_class(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=num_image_text_embeds,
                    dtype=self.dtype,
                )

                value_dict = {}
                for i, state_dict in enumerate(state_dicts):
                    value_dict.update({f"to_k_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
                    value_dict.update({f"to_v_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})
                    value_dict.update({f"to_k_ip.{i}.bias": state_dict["ip_adapter"][f"{key_id}.to_k_ip.bias"]})
                    value_dict.update({f"to_v_ip.{i}.bias": state_dict["ip_adapter"][f"{key_id}.to_v_ip.bias"]})

                _load_state_dict_into_model(attn_procs[name], value_dict)

                key_id += 1

        return attn_procs

    def _load_ip_adapter_weights(self, state_dicts, low_cpu_mem_usage=False):
        if not isinstance(state_dicts, list):
            state_dicts = [state_dicts]

        self.encoder_hid_proj = None

        attn_procs = self._convert_ip_adapter_attn_to_diffusers(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)
        self.set_attn_processor(attn_procs)

        image_projection_layers = []
        for state_dict in state_dicts:
            image_projection_layer = self._convert_ip_adapter_image_proj_to_diffusers(
                state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
            )
            image_projection_layers.append(image_projection_layer)

        self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        self.config.encoder_hid_dim_type = "ip_image_proj"
