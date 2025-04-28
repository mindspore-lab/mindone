# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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
from ...loaders import PeftAdapterMixin
from ...models.attention_processor import AttentionProcessor
from ...models.modeling_utils import ModelMixin
from ...utils import BaseOutput, logging
from ..controlnets.controlnet import ControlNetConditioningEmbedding, zero_module
from ..embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from ..modeling_outputs import Transformer2DModelOutput
from ..transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FluxControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[ms.Tensor]
    controlnet_single_block_samples: Tuple[ms.Tensor]


class FluxControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: List[int] = [16, 56, 56],
        num_mode: int = None,
        conditioning_embedding_channels: int = None,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = mint.nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = mint.nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.CellList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.CellList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_single_layers)
            ]
        )

        # controlnet_blocks
        controlnet_blocks = []
        for _ in range(len(self.transformer_blocks)):
            controlnet_blocks.append(zero_module(mint.nn.Linear(self.inner_dim, self.inner_dim)))  # zero_module
        self.controlnet_blocks = nn.CellList(controlnet_blocks)

        controlnet_single_blocks = []
        for _ in range(len(self.single_transformer_blocks)):
            controlnet_single_blocks.append(zero_module(mint.nn.Linear(self.inner_dim, self.inner_dim)))  # zero_module
        self.controlnet_single_blocks = nn.CellList(controlnet_single_blocks)

        self.union = num_mode is not None
        if self.union:
            self.controlnet_mode_embedder = mint.nn.Embedding(num_mode, self.inner_dim)

        if conditioning_embedding_channels is not None:
            self.input_hint_block = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=conditioning_embedding_channels, block_out_channels=(16, 16, 16, 16)
            )
            self.controlnet_x_embedder = mint.nn.Linear(in_channels, self.inner_dim)
        else:
            self.input_hint_block = None
            self.controlnet_x_embedder = zero_module(mint.nn.Linear(in_channels, self.inner_dim))  # zero_module

        self.gradient_checkpointing = False
        self.patch_size = self.config.patch_size

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

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: int = 4,
        num_single_layers: int = 10,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        load_weights_from_transformer=True,
    ):
        config = dict(transformer.config)
        config["num_layers"] = num_layers
        config["num_single_layers"] = num_single_layers
        config["attention_head_dim"] = attention_head_dim
        config["num_attention_heads"] = num_attention_heads

        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            ms.load_param_into_net(controlnet.pos_embed, transformer.pos_embed.parameters_dict())
            ms.load_param_into_net(controlnet.time_text_embed, transformer.time_text_embed.parameters_dict())
            ms.load_param_into_net(controlnet.context_embedder, transformer.context_embedder.parameters_dict())
            ms.load_param_into_net(controlnet.x_embedder, transformer.x_embedder.parameters_dict())
            ms.load_param_into_net(
                controlnet.transformer_blocks, transformer.transformer_blocks.parameters_dict(), strict_load=False
            )
            ms.load_param_into_net(
                controlnet.single_transformer_blocks,
                transformer.single_transformer_blocks.parameters_dict(),
                strict_load=False,
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
        hidden_states: ms.Tensor,
        controlnet_cond: ms.Tensor,
        controlnet_mode: ms.Tensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: ms.Tensor = None,
        pooled_projections: ms.Tensor = None,
        timestep: ms.Tensor = None,
        img_ids: ms.Tensor = None,
        txt_ids: ms.Tensor = None,
        guidance: ms.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[ms.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`ms.Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            controlnet_cond (`ms.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            controlnet_mode (`ms.Tensor`):
                The mode tensor of shape `(batch_size, 1)`.
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
            return_dict (`bool`, *optional*, defaults to `False`):
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
        hidden_states = self.x_embedder(hidden_states)

        if self.input_hint_block is not None:
            controlnet_cond = self.input_hint_block(controlnet_cond)
            batch_size, channels, height_pw, width_pw = controlnet_cond.shape
            height = height_pw // self.patch_size
            width = width_pw // self.patch_size
            controlnet_cond = controlnet_cond.reshape(
                batch_size, channels, height, self.patch_size, width, self.patch_size
            )
            controlnet_cond = controlnet_cond.permute(0, 2, 4, 1, 3, 5)
            controlnet_cond = controlnet_cond.reshape(batch_size, height * width, -1)
        # add
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if self.union:
            # union mode
            if controlnet_mode is None:
                raise ValueError("`controlnet_mode` cannot be `None` when applying ControlNet-Union")
            # union mode emb
            controlnet_mode_emb = self.controlnet_mode_embedder(controlnet_mode)
            encoder_hidden_states = mint.cat([controlnet_mode_emb, encoder_hidden_states], dim=1)
            txt_ids = mint.cat([txt_ids[:1], txt_ids], dim=0)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d ms.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d ms.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = mint.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        block_samples = ()
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            block_samples = block_samples + (hidden_states,)

        hidden_states = mint.cat([encoder_hidden_states, hidden_states], dim=1)

        single_block_samples = ()
        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            single_block_samples = single_block_samples + (hidden_states[:, encoder_hidden_states.shape[1] :],)

        # controlnet block
        controlnet_block_samples = ()
        for block_sample, controlnet_block in zip(block_samples, self.controlnet_blocks):
            block_sample = controlnet_block(block_sample)
            controlnet_block_samples = controlnet_block_samples + (block_sample,)

        controlnet_single_block_samples = ()
        for single_block_sample, controlnet_block in zip(single_block_samples, self.controlnet_single_blocks):
            single_block_sample = controlnet_block(single_block_sample)
            controlnet_single_block_samples = controlnet_single_block_samples + (single_block_sample,)

        # scaling
        controlnet_block_samples = [sample * conditioning_scale for sample in controlnet_block_samples]
        controlnet_single_block_samples = [sample * conditioning_scale for sample in controlnet_single_block_samples]

        controlnet_block_samples = None if len(controlnet_block_samples) == 0 else controlnet_block_samples
        controlnet_single_block_samples = (
            None if len(controlnet_single_block_samples) == 0 else controlnet_single_block_samples
        )

        if not return_dict:
            return (controlnet_block_samples, controlnet_single_block_samples)

        return FluxControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )


class FluxMultiControlNetModel(ModelMixin):
    r"""
    `FluxMultiControlNetModel` wrapper class for Multi-FluxControlNetModel

    This module is a wrapper for multiple instances of the `FluxControlNetModel`. The `forward()` API is designed to be
    compatible with `FluxControlNetModel`.

    Args:
        controlnets (`List[FluxControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `FluxControlNetModel` as a list.
    """

    def __init__(self, controlnets):
        super().__init__()
        self.nets = nn.CellList(controlnets)

    def construct(
        self,
        hidden_states: ms.Tensor,
        controlnet_cond: List[ms.Tensor],
        controlnet_mode: List[ms.Tensor],
        conditioning_scale: List[float],
        encoder_hidden_states: ms.Tensor = None,
        pooled_projections: ms.Tensor = None,
        timestep: ms.Tensor = None,
        img_ids: ms.Tensor = None,
        txt_ids: ms.Tensor = None,
        guidance: ms.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[FluxControlNetOutput, Tuple]:
        # ControlNet-Union with multiple conditions
        # only load one ControlNet for saving memories
        if len(self.nets) == 1 and self.nets[0].union:
            controlnet = self.nets[0]

            for i, (image, mode, scale) in enumerate(zip(controlnet_cond, controlnet_mode, conditioning_scale)):
                block_samples, single_block_samples = controlnet(
                    hidden_states=hidden_states,
                    controlnet_cond=image,
                    controlnet_mode=mode[:, None],
                    conditioning_scale=scale,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_projections,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=return_dict,
                )

                # merge samples
                if i == 0:
                    control_block_samples = block_samples
                    control_single_block_samples = single_block_samples
                else:
                    control_block_samples = [
                        control_block_sample + block_sample
                        for control_block_sample, block_sample in zip(control_block_samples, block_samples)
                    ]

                    control_single_block_samples = [
                        control_single_block_sample + block_sample
                        for control_single_block_sample, block_sample in zip(
                            control_single_block_samples, single_block_samples
                        )
                    ]

        # Regular Multi-ControlNets
        # load all ControlNets into memories
        else:
            for i, (image, mode, scale, controlnet) in enumerate(
                zip(controlnet_cond, controlnet_mode, conditioning_scale, self.nets)
            ):
                block_samples, single_block_samples = controlnet(
                    hidden_states=hidden_states,
                    controlnet_cond=image,
                    controlnet_mode=mode[:, None],
                    conditioning_scale=scale,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_projections,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=return_dict,
                )

                # merge samples
                if i == 0:
                    control_block_samples = block_samples
                    control_single_block_samples = single_block_samples
                else:
                    if block_samples is not None and control_block_samples is not None:
                        control_block_samples = [
                            control_block_sample + block_sample
                            for control_block_sample, block_sample in zip(control_block_samples, block_samples)
                        ]
                    if single_block_samples is not None and control_single_block_samples is not None:
                        control_single_block_samples = [
                            control_single_block_sample + block_sample
                            for control_single_block_sample, block_sample in zip(
                                control_single_block_samples, single_block_samples
                            )
                        ]

        return control_block_samples, control_single_block_samples
