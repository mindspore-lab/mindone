# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mindspore as ms
from mindspore import mint, nn
from mindspore.common.initializer import initializer

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import BaseOutput, logging
from ..attention_processor import AttentionProcessor
from ..cache_utils import CacheMixin
from ..controlnets.controlnet import zero_module
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..transformers.transformer_qwenimage import (
    QwenEmbedRope,
    QwenImageTransformerBlock,
    QwenTimestepProjEmbeddings,
    RMSNorm,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class QwenImageControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[ms.tensor]


class QwenImageControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        extra_condition_channels: int = 0,  # for controlnet-inpainting
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = mint.nn.Linear(in_channels, self.inner_dim)
        self.txt_in = mint.nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.CellList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # controlnet_blocks
        controlnet_blocks = []
        for _ in range(len(self.transformer_blocks)):
            controlnet_blocks.append(zero_module(mint.nn.Linear(self.inner_dim, self.inner_dim)))
        self.controlnet_x_embedder = zero_module(mint.nn.Linear(in_channels + extra_condition_channels, self.inner_dim))
        self.controlnet_blocks = nn.CellList(controlnet_blocks)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self):
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
    def set_attn_processor(self, processor):
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

    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: int = 5,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        load_weights_from_transformer=True,
        extra_condition_channels: int = 0,
    ):
        config = dict(transformer.config)
        config["num_layers"] = num_layers
        config["attention_head_dim"] = attention_head_dim
        config["num_attention_heads"] = num_attention_heads
        config["extra_condition_channels"] = extra_condition_channels

        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            ms.load_param_into_net(controlnet.pos_embed, transformer.pos_embed.parameters_dict())
            ms.load_param_into_net(controlnet.time_text_embed, transformer.time_text_embed.parameters_dict())
            ms.load_param_into_net(controlnet.img_in, transformer.img_in.parameters_dict())
            ms.load_param_into_net(controlnet.txt_in, transformer.txt_in.parameters_dict())
            ms.load_param_into_net(
                controlnet.transformer_blocks, transformer.transformer_blocks.parameters_dict(), strict_load=False
            )

            # zero_module
            controlnet.controlnet_x_embedder.weight.set_data(
                initializer(
                    "zeros",
                    controlnet.controlnet_x_embedder.weight.shape,
                    controlnet.controlnet_x_embedder.weight.dtype,
                )
            )
            controlnet.controlnet_x_embedder.bias.set_data(
                initializer(
                    "zeros", controlnet.controlnet_x_embedder.bias.shape, controlnet.controlnet_x_embedder.bias.dtype
                )
            )

        return controlnet

    def construct(
        self,
        hidden_states: ms.tensor,
        controlnet_cond: ms.tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: ms.tensor = None,
        encoder_hidden_states_mask: ms.tensor = None,
        timestep: ms.tensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[ms.tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

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
            block_controlnet_hidden_states: (`list` of `ms.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()

        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )
        hidden_states = self.img_in(hidden_states)

        # add
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        temb = self.time_text_embed(timestep, hidden_states)

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        block_samples = ()
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            block_samples = block_samples + (hidden_states,)

        # controlnet block
        controlnet_block_samples = ()
        for block_sample, controlnet_block in zip(block_samples, self.controlnet_blocks):
            block_sample = controlnet_block(block_sample)
            controlnet_block_samples = controlnet_block_samples + (block_sample,)

        # scaling
        controlnet_block_samples = [sample * conditioning_scale for sample in controlnet_block_samples]
        controlnet_block_samples = None if len(controlnet_block_samples) == 0 else controlnet_block_samples

        if not return_dict:
            return controlnet_block_samples

        return QwenImageControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
        )


class QwenImageMultiControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    `QwenImageMultiControlNetModel` wrapper class for Multi-QwenImageControlNetModel

    This module is a wrapper for multiple instances of the `QwenImageControlNetModel`. The `construct()` API is designed
    to be compatible with `QwenImageControlNetModel`.

    Args:
        controlnets (`List[QwenImageControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `QwenImageControlNetModel` as a list.
    """

    def __init__(self, controlnets):
        super().__init__()
        self.nets = nn.CellList(controlnets)

    def construct(
        self,
        hidden_states: ms.tensor,
        controlnet_cond: List[ms.tensor],
        conditioning_scale: List[float],
        encoder_hidden_states: ms.tensor = None,
        encoder_hidden_states_mask: ms.tensor = None,
        timestep: ms.tensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[QwenImageControlNetOutput, Tuple]:
        # ControlNet-Union with multiple conditions
        # only load one ControlNet for saving memories
        if len(self.nets) == 1:
            controlnet = self.nets[0]

            for i, (image, scale) in enumerate(zip(controlnet_cond, conditioning_scale)):
                block_samples = controlnet(
                    hidden_states=hidden_states,
                    controlnet_cond=image,
                    conditioning_scale=scale,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    timestep=timestep,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=return_dict,
                )

                # merge samples
                if i == 0:
                    control_block_samples = block_samples
                else:
                    if block_samples is not None and control_block_samples is not None:
                        control_block_samples = [
                            control_block_sample + block_sample
                            for control_block_sample, block_sample in zip(control_block_samples, block_samples)
                        ]
        else:
            raise ValueError("QwenImageMultiControlNetModel only supports a single controlnet-union now.")

        return control_block_samples
