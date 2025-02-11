# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
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


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mindspore as ms
from mindspore import nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..loaders import PeftAdapterMixin
from ..models.attention import JointTransformerBlock
from ..models.attention_processor import AttentionProcessor
from ..models.modeling_utils import ModelMixin
from ..utils import logging
from .controlnet import BaseOutput
from .embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from .transformers.transformer_2d import Transformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class SD3ControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[ms.Tensor]


class SD3ControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = False  # not supported now

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        self.context_embedder = nn.Dense(joint_attention_dim, caption_projection_dim)

        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        self.transformer_blocks = nn.CellList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    context_pre_only=False,
                )
                for i in range(num_layers)
            ]
        )

        # controlnet_blocks
        self.controlnet_blocks = []
        for _ in range(len(self.transformer_blocks)):
            controlnet_block = nn.Dense(
                self.inner_dim,
                self.inner_dim,
                weight_init="zeros",
                bias_init="zeros",
            )  # zero_module
            self.controlnet_blocks.append(controlnet_block)
        self.controlnet_blocks = nn.CellList(self.controlnet_blocks)

        self.pos_embed_input = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_type=None,
            zero_module=True,
        )  # zero module, FIXME: only conv2d zero

        self.gradient_checkpointing = False

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: nn.Cell, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.name_cells().values():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.name_cells().values():
            fn_recursive_feed_forward(module, chunk_size, dim)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: nn.Cell, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.name_cells().items():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.name_cells().items():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: nn.Cell, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.name_cells().items():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.name_cells().items():
            fn_recursive_attn_processor(name, module, processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @classmethod
    def from_transformer(cls, transformer, num_layers=12, load_weights_from_transformer=True):
        config = transformer.config
        config["num_layers"] = num_layers or config.num_layers
        controlnet = cls(**config)

        if load_weights_from_transformer:
            ms.load_param_into_net(controlnet.pos_embed, transformer.pos_embed.parameters_dict())
            ms.load_param_into_net(controlnet.time_text_embed, transformer.time_text_embed.parameters_dict())
            ms.load_param_into_net(controlnet.context_embedder, transformer.context_embedder.parameters_dict())
            ms.load_param_into_net(controlnet.transformer_blocks, transformer.transformer_blocks.parameters_dict())

            # No `zero_module` here for it is done in cls.__init__
            # controlnet.pos_embed_input = zero_module(controlnet.pos_embed_input)

        return controlnet

    def construct(
        self,
        hidden_states: ms.Tensor,
        controlnet_cond: ms.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: ms.Tensor = None,
        pooled_projections: ms.Tensor = None,
        timestep: ms.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[ms.Tensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`ms.Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            controlnet_cond (`ms.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            encoder_hidden_states (`ms.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`ms.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `ms.Tensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            # weight the lora layers by setting `lora_scale` for each PEFT layer here
            # and remove `lora_scale` from each PEFT layer at the end.
            # scale_lora_layers & unscale_lora_layers maybe contains some operation forbidden in graph mode
            raise RuntimeError(
                f"You are trying to set scaling of lora layer by passing {joint_attention_kwargs['scale']}. "
                f"However it's not allowed in on-the-fly model forwarding. "
                f"Please manually call `scale_lora_layers(model, lora_scale)` before model forwarding and "
                f"`unscale_lora_layers(model, lora_scale)` after model forwarding. "
                f"For example, it can be done in a pipeline call like `StableDiffusionPipeline.__call__`."
            )

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # add
        hidden_states = hidden_states + self.pos_embed_input(controlnet_cond)

        block_res_samples = ()

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
            )

            block_res_samples = block_res_samples + (hidden_states,)

        controlnet_block_res_samples = ()
        for block_res_sample, controlnet_block in zip(block_res_samples, self.controlnet_blocks):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples = controlnet_block_res_samples + (block_res_sample,)

        # 6. scaling
        controlnet_block_res_samples = [sample * conditioning_scale for sample in controlnet_block_res_samples]

        if not return_dict:
            return (controlnet_block_res_samples,)

        return SD3ControlNetOutput(controlnet_block_samples=controlnet_block_res_samples)


class SD3MultiControlNetModel(ModelMixin):
    r"""
    `SD3ControlNetModel` wrapper class for Multi-SD3ControlNet

    This module is a wrapper for multiple instances of the `SD3ControlNetModel`. The `forward()` API is designed to be
    compatible with `SD3ControlNetModel`.

    Args:
        controlnets (`List[SD3ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `SD3ControlNetModel` as a list.
    """

    def __init__(self, controlnets):
        super().__init__()
        self.nets = nn.CellList(controlnets)

    def construct(
        self,
        hidden_states: ms.Tensor,
        controlnet_cond: List[ms.Tensor],
        conditioning_scale: List[float],
        pooled_projections: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        timestep: ms.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[SD3ControlNetOutput, Tuple]:
        # adapt to graph mode
        control_block_samples = None

        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
            block_samples = controlnet(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                controlnet_cond=image,
                conditioning_scale=scale,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=return_dict,
            )

            # merge samples
            if i == 0:
                control_block_samples = block_samples
            else:
                control_block_samples = [
                    control_block_sample + block_sample
                    for control_block_sample, block_sample in zip(control_block_samples[0], block_samples[0])
                ]
                control_block_samples = (tuple(control_block_samples),)

        return control_block_samples
