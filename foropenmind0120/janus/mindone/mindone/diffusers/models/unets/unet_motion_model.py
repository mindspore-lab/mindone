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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mindspore as ms
from mindspore import nn, ops

from ...configuration_utils import ConfigMixin, FrozenDict, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin, UNet2DConditionLoadersMixin
from ...utils import BaseOutput, logging
from ..attention import BasicTransformerBlock
from ..attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor, IPAdapterAttnProcessor
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from ..normalization import GroupNorm
from ..resnet import Downsample2D, ResnetBlock2D, Upsample2D
from ..transformers.dual_transformer_2d import DualTransformer2DModel
from ..transformers.transformer_2d import Transformer2DModel
from .unet_2d_blocks import UNetMidBlock2DCrossAttn
from .unet_2d_condition import UNet2DConditionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNetMotionOutput(BaseOutput):
    """
    The output of [`UNetMotionOutput`].

    Args:
        sample (`ms.Tensor` of shape `(batch_size, num_channels, num_frames, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: ms.Tensor


class AnimateDiffTransformer3D(nn.Cell):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward. See `diffusers.models.activations.get_activation` for supported
            activation functions.
        norm_elementwise_affine (`bool`, *optional*):
            Configure if the `TransformerBlock` should use learnable elementwise affine parameters for normalization.
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
        positional_embeddings: (`str`, *optional*):
            The type of positional embeddings to apply to the sequence input before passing use.
        num_positional_embeddings: (`int`, *optional*):
            The maximum length of the sequence over which to apply positional embeddings.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Dense(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    double_self_attention=double_self_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=num_positional_embeddings,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = nn.Dense(inner_dim, in_channels)

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        timestep: Optional[ms.Tensor] = None,
        class_labels: Optional[ms.Tensor] = None,
        num_frames: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ms.Tensor:
        """
        The [`AnimateDiffTransformer3D`] forward method.

        Args:
            hidden_states (`ms.Tensor` of shape `(batch size, num latent pixels)` if discrete, `ms.Tensor` of shape `(batch size, channel, height, width)` if continuous):  # noqa: E501
                Input hidden_states.
            encoder_hidden_states ( `ms.Tensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `ms.Tensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `ms.Tensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            num_frames (`int`, *optional*, defaults to 1):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Returns:
            ms.Tensor:
                The output tensor.
        """
        # 1. Input
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        residual = hidden_states

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)

        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states[None, None, :]
            .reshape(batch_size, height, width, num_frames, channel)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)

        output = hidden_states + residual
        return output


class DownBlockMotion(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        temporal_num_attention_heads: Union[int, Tuple[int]] = 1,
        temporal_cross_attention_dim: Optional[int] = None,
        temporal_max_seq_length: int = 32,
        temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        temporal_double_self_attention: bool = True,
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        # support for variable transformer layers per temporal block
        if isinstance(temporal_transformer_layers_per_block, int):
            temporal_transformer_layers_per_block = (temporal_transformer_layers_per_block,) * num_layers
        elif len(temporal_transformer_layers_per_block) != num_layers:
            raise ValueError(
                f"`temporal_transformer_layers_per_block` must be an integer or a tuple of integers of length {num_layers}"
            )

        # support for variable number of attention head per temporal layers
        if isinstance(temporal_num_attention_heads, int):
            temporal_num_attention_heads = (temporal_num_attention_heads,) * num_layers
        elif len(temporal_num_attention_heads) != num_layers:
            raise ValueError(
                f"`temporal_num_attention_heads` must be an integer or a tuple of integers of length {num_layers}"
            )

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            motion_modules.append(
                AnimateDiffTransformer3D(
                    num_attention_heads=temporal_num_attention_heads[i],
                    in_channels=out_channels,
                    num_layers=temporal_transformer_layers_per_block[i],
                    norm_num_groups=resnet_groups,
                    cross_attention_dim=temporal_cross_attention_dim,
                    attention_bias=False,
                    activation_fn="geglu",
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=temporal_max_seq_length,
                    attention_head_dim=out_channels // temporal_num_attention_heads[i],
                    double_self_attention=temporal_double_self_attention,
                )
            )

        self.resnets = nn.CellList(resnets)
        self.motion_modules = nn.CellList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.CellList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False
        self.has_cross_attention = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        temb: Optional[ms.Tensor] = None,
        num_frames: int = 1,
    ) -> Union[ms.Tensor, Tuple[ms.Tensor, ...]]:
        output_states = ()

        blocks = zip(self.resnets, self.motion_modules)
        for resnet, motion_module in blocks:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = motion_module(hidden_states, num_frames=num_frames)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlockMotion(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        temporal_cross_attention_dim: Optional[int] = None,
        temporal_num_attention_heads: int = 8,
        temporal_max_seq_length: int = 32,
        temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        temporal_double_self_attention: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        motion_modules = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        # support for variable transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = (transformer_layers_per_block,) * num_layers
        elif len(transformer_layers_per_block) != num_layers:
            raise ValueError(
                f"transformer_layers_per_block must be an integer or a list of integers of length {num_layers}"
            )

        # support for variable transformer layers per temporal block
        if isinstance(temporal_transformer_layers_per_block, int):
            temporal_transformer_layers_per_block = (temporal_transformer_layers_per_block,) * num_layers
        elif len(temporal_transformer_layers_per_block) != num_layers:
            raise ValueError(
                f"temporal_transformer_layers_per_block must be an integer or a list of integers of length {num_layers}"
            )

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

            motion_modules.append(
                AnimateDiffTransformer3D(
                    num_attention_heads=temporal_num_attention_heads,
                    in_channels=out_channels,
                    num_layers=temporal_transformer_layers_per_block[i],
                    norm_num_groups=resnet_groups,
                    cross_attention_dim=temporal_cross_attention_dim,
                    attention_bias=False,
                    activation_fn="geglu",
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=temporal_max_seq_length,
                    attention_head_dim=out_channels // temporal_num_attention_heads,
                    double_self_attention=temporal_double_self_attention,
                )
            )

        self.attentions = nn.CellList(attentions)
        self.resnets = nn.CellList(resnets)
        self.motion_modules = nn.CellList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.CellList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        temb: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        num_frames: int = 1,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        additional_residuals: Optional[ms.Tensor] = None,
    ):
        output_states = ()

        blocks = list(zip(self.resnets, self.attentions, self.motion_modules))
        for i, (resnet, attn, motion_module) in enumerate(blocks):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
            hidden_states = motion_module(
                hidden_states,
                num_frames=num_frames,
            )

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlockMotion(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        temporal_cross_attention_dim: Optional[int] = None,
        temporal_num_attention_heads: int = 8,
        temporal_max_seq_length: int = 32,
        temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,
    ):
        super().__init__()
        resnets = []
        attentions = []
        motion_modules = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        # support for variable transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = (transformer_layers_per_block,) * num_layers
        elif len(transformer_layers_per_block) != num_layers:
            raise ValueError(
                f"transformer_layers_per_block must be an integer or a list of integers of length {num_layers}, got {len(transformer_layers_per_block)}"
            )

        # support for variable transformer layers per temporal block
        if isinstance(temporal_transformer_layers_per_block, int):
            temporal_transformer_layers_per_block = (temporal_transformer_layers_per_block,) * num_layers
        elif len(temporal_transformer_layers_per_block) != num_layers:
            raise ValueError(
                f"temporal_transformer_layers_per_block must be an integer or a list of integers of length {num_layers}, got {len(temporal_transformer_layers_per_block)}"  # noqa: E501
            )

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            motion_modules.append(
                AnimateDiffTransformer3D(
                    num_attention_heads=temporal_num_attention_heads,
                    in_channels=out_channels,
                    num_layers=temporal_transformer_layers_per_block[i],
                    norm_num_groups=resnet_groups,
                    cross_attention_dim=temporal_cross_attention_dim,
                    attention_bias=False,
                    activation_fn="geglu",
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=temporal_max_seq_length,
                    attention_head_dim=out_channels // temporal_num_attention_heads,
                )
            )

        self.attentions = nn.CellList(attentions)
        self.resnets = nn.CellList(resnets)
        self.motion_modules = nn.CellList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.CellList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def construct(
        self,
        hidden_states: ms.Tensor,
        res_hidden_states_tuple: Tuple[ms.Tensor, ...],
        temb: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        num_frames: int = 1,
    ) -> ms.Tensor:
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        blocks = zip(self.resnets, self.attentions, self.motion_modules)
        for resnet, attn, motion_module in blocks:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                raise NotImplementedError("apply_freeu is not implemented")

            hidden_states = ops.cat([hidden_states, res_hidden_states], axis=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

            hidden_states = motion_module(
                hidden_states,
                num_frames=num_frames,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlockMotion(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temporal_cross_attention_dim: Optional[int] = None,
        temporal_num_attention_heads: int = 8,
        temporal_max_seq_length: int = 32,
        temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        # support for variable transformer layers per temporal block
        if isinstance(temporal_transformer_layers_per_block, int):
            temporal_transformer_layers_per_block = (temporal_transformer_layers_per_block,) * num_layers
        elif len(temporal_transformer_layers_per_block) != num_layers:
            raise ValueError(
                f"temporal_transformer_layers_per_block must be an integer or a list of integers of length {num_layers}"
            )

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            motion_modules.append(
                AnimateDiffTransformer3D(
                    num_attention_heads=temporal_num_attention_heads,
                    in_channels=out_channels,
                    num_layers=temporal_transformer_layers_per_block[i],
                    norm_num_groups=resnet_groups,
                    cross_attention_dim=temporal_cross_attention_dim,
                    attention_bias=False,
                    activation_fn="geglu",
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=temporal_max_seq_length,
                    attention_head_dim=out_channels // temporal_num_attention_heads,
                )
            )

        self.resnets = nn.CellList(resnets)
        self.motion_modules = nn.CellList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.CellList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx
        self.has_cross_attention = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        res_hidden_states_tuple: Tuple[ms.Tensor, ...],
        temb: Optional[ms.Tensor] = None,
        upsample_size=None,
        num_frames: int = 1,
    ) -> ms.Tensor:
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        blocks = zip(self.resnets, self.motion_modules)

        for resnet, motion_module in blocks:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                raise NotImplementedError("apply_freeu is not implemented")

            hidden_states = ops.cat([hidden_states, res_hidden_states], axis=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = motion_module(hidden_states, num_frames=num_frames)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UNetMidBlockCrossAttnMotion(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        temporal_num_attention_heads: int = 1,
        temporal_cross_attention_dim: Optional[int] = None,
        temporal_max_seq_length: int = 32,
        temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.has_motion_modules = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # support for variable transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = (transformer_layers_per_block,) * num_layers
        elif len(transformer_layers_per_block) != num_layers:
            raise ValueError(
                f"`transformer_layers_per_block` should be an integer or a list of integers of length {num_layers}."
            )

        # support for variable transformer layers per temporal block
        if isinstance(temporal_transformer_layers_per_block, int):
            temporal_transformer_layers_per_block = (temporal_transformer_layers_per_block,) * num_layers
        elif len(temporal_transformer_layers_per_block) != num_layers:
            raise ValueError(
                f"`temporal_transformer_layers_per_block` should be an integer or a list of integers of length {num_layers}."
            )

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        motion_modules = []

        for i in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            motion_modules.append(
                AnimateDiffTransformer3D(
                    num_attention_heads=temporal_num_attention_heads,
                    attention_head_dim=in_channels // temporal_num_attention_heads,
                    in_channels=in_channels,
                    num_layers=temporal_transformer_layers_per_block[i],
                    norm_num_groups=resnet_groups,
                    cross_attention_dim=temporal_cross_attention_dim,
                    attention_bias=False,
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=temporal_max_seq_length,
                    activation_fn="geglu",
                )
            )

        self.attentions = nn.CellList(attentions)
        self.resnets = nn.CellList(resnets)
        self.motion_modules = nn.CellList(motion_modules)

        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        temb: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        num_frames: int = 1,
    ) -> ms.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)

        blocks = zip(self.attentions, self.resnets[1:], self.motion_modules)
        for attn, resnet, motion_module in blocks:
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
            hidden_states = motion_module(
                hidden_states,
                num_frames=num_frames,
            )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class MotionModules(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        layers_per_block: int = 2,
        transformer_layers_per_block: Union[int, Tuple[int]] = 8,
        num_attention_heads: Union[int, Tuple[int]] = 8,
        attention_bias: bool = False,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_num_groups: int = 32,
        max_seq_length: int = 32,
    ):
        super().__init__()
        self.motion_modules = []

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = (transformer_layers_per_block,) * layers_per_block
        elif len(transformer_layers_per_block) != layers_per_block:
            raise ValueError(
                f"The number of transformer layers per block must match the number of layers per block, "
                f"got {layers_per_block} and {len(transformer_layers_per_block)}"
            )

        for i in range(layers_per_block):
            self.motion_modules.append(
                AnimateDiffTransformer3D(
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block[i],
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=in_channels // num_attention_heads,
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=max_seq_length,
                )
            )
        self.motion_modules = nn.CellList(self.motion_modules)


class MotionAdapter(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
        self,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        motion_layers_per_block: Union[int, Tuple[int]] = 2,
        motion_transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple[int]]] = 1,
        motion_mid_block_layers_per_block: int = 1,
        motion_transformer_layers_per_mid_block: Union[int, Tuple[int]] = 1,
        motion_num_attention_heads: Union[int, Tuple[int]] = 8,
        motion_norm_num_groups: int = 32,
        motion_max_seq_length: int = 32,
        use_motion_mid_block: bool = True,
        conv_in_channels: Optional[int] = None,
    ):
        """Container to store AnimateDiff Motion Modules

        Args:
            block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each UNet block.
            motion_layers_per_block (`int` or `Tuple[int]`, *optional*, defaults to 2):
                The number of motion layers per UNet block.
            motion_transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple[int]]`, *optional*, defaults to 1):
                The number of transformer layers to use in each motion layer in each block.
            motion_mid_block_layers_per_block (`int`, *optional*, defaults to 1):
                The number of motion layers in the middle UNet block.
            motion_transformer_layers_per_mid_block (`int` or `Tuple[int]`, *optional*, defaults to 1):
                The number of transformer layers to use in each motion layer in the middle block.
            motion_num_attention_heads (`int` or `Tuple[int]`, *optional*, defaults to 8):
                The number of heads to use in each attention layer of the motion module.
            motion_norm_num_groups (`int`, *optional*, defaults to 32):
                The number of groups to use in each group normalization layer of the motion module.
            motion_max_seq_length (`int`, *optional*, defaults to 32):
                The maximum sequence length to use in the motion module.
            use_motion_mid_block (`bool`, *optional*, defaults to True):
                Whether to use a motion module in the middle of the UNet.
        """

        super().__init__()
        down_blocks = []
        up_blocks = []

        if isinstance(motion_layers_per_block, int):
            motion_layers_per_block = (motion_layers_per_block,) * len(block_out_channels)
        elif len(motion_layers_per_block) != len(block_out_channels):
            raise ValueError(
                f"The number of motion layers per block must match the number of blocks, "
                f"got {len(block_out_channels)} and {len(motion_layers_per_block)}"
            )

        if isinstance(motion_transformer_layers_per_block, int):
            motion_transformer_layers_per_block = (motion_transformer_layers_per_block,) * len(block_out_channels)

        if isinstance(motion_transformer_layers_per_mid_block, int):
            motion_transformer_layers_per_mid_block = (
                motion_transformer_layers_per_mid_block,
            ) * motion_mid_block_layers_per_block
        elif len(motion_transformer_layers_per_mid_block) != motion_mid_block_layers_per_block:
            raise ValueError(
                f"The number of layers per mid block ({motion_mid_block_layers_per_block}) "
                f"must match the length of motion_transformer_layers_per_mid_block ({len(motion_transformer_layers_per_mid_block)})"
            )

        if isinstance(motion_num_attention_heads, int):
            motion_num_attention_heads = (motion_num_attention_heads,) * len(block_out_channels)
        elif len(motion_num_attention_heads) != len(block_out_channels):
            raise ValueError(
                f"The length of the attention head number tuple in the motion module must match the "
                f"number of block, got {len(motion_num_attention_heads)} and {len(block_out_channels)}"
            )

        if conv_in_channels:
            # input
            self.conv_in = nn.Conv2d(
                conv_in_channels, block_out_channels[0], kernel_size=3, pad_mode="pad", padding=1, has_bias=True
            )
        else:
            self.conv_in = None

        for i, channel in enumerate(block_out_channels):
            output_channel = block_out_channels[i]
            down_blocks.append(
                MotionModules(
                    in_channels=output_channel,
                    norm_num_groups=motion_norm_num_groups,
                    cross_attention_dim=None,
                    activation_fn="geglu",
                    attention_bias=False,
                    num_attention_heads=motion_num_attention_heads[i],
                    max_seq_length=motion_max_seq_length,
                    layers_per_block=motion_layers_per_block[i],
                    transformer_layers_per_block=motion_transformer_layers_per_block[i],
                )
            )

        if use_motion_mid_block:
            self.mid_block = MotionModules(
                in_channels=block_out_channels[-1],
                norm_num_groups=motion_norm_num_groups,
                cross_attention_dim=None,
                activation_fn="geglu",
                attention_bias=False,
                num_attention_heads=motion_num_attention_heads[-1],
                max_seq_length=motion_max_seq_length,
                layers_per_block=motion_mid_block_layers_per_block,
                transformer_layers_per_block=motion_transformer_layers_per_mid_block,
            )
        else:
            self.mid_block = None

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        reversed_motion_layers_per_block = list(reversed(motion_layers_per_block))
        reversed_motion_transformer_layers_per_block = list(reversed(motion_transformer_layers_per_block))
        reversed_motion_num_attention_heads = list(reversed(motion_num_attention_heads))
        for i, channel in enumerate(reversed_block_out_channels):
            output_channel = reversed_block_out_channels[i]
            up_blocks.append(
                MotionModules(
                    in_channels=output_channel,
                    norm_num_groups=motion_norm_num_groups,
                    cross_attention_dim=None,
                    activation_fn="geglu",
                    attention_bias=False,
                    num_attention_heads=reversed_motion_num_attention_heads[i],
                    max_seq_length=motion_max_seq_length,
                    layers_per_block=reversed_motion_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_motion_transformer_layers_per_block[i],
                )
            )

        self.down_blocks = nn.CellList(down_blocks)
        self.up_blocks = nn.CellList(up_blocks)

    def construct(self, sample):
        pass


class UNetMotionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin):
    r"""
    A modified conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a
    sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlockMotion",
            "CrossAttnDownBlockMotion",
            "CrossAttnDownBlockMotion",
            "DownBlockMotion",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlockMotion",
            "CrossAttnUpBlockMotion",
            "CrossAttnUpBlockMotion",
            "CrossAttnUpBlockMotion",
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Union[int, Tuple[int], Tuple[Tuple]]] = None,
        temporal_transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_temporal_transformer_layers_per_block: Optional[Union[int, Tuple[int], Tuple[Tuple]]] = None,
        transformer_layers_per_mid_block: Optional[Union[int, Tuple[int]]] = None,
        temporal_transformer_layers_per_mid_block: Optional[Union[int, Tuple[int]]] = 1,
        use_linear_projection: bool = False,
        num_attention_heads: Union[int, Tuple[int, ...]] = 8,
        motion_max_seq_length: int = 32,
        motion_num_attention_heads: Union[int, Tuple[int, ...]] = 8,
        reverse_motion_num_attention_heads: Optional[Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]]] = None,
        use_motion_mid_block: bool = True,
        mid_block_layers: int = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        projection_class_embeddings_input_dim: Optional[int] = None,
        time_cond_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        self.sample_size = sample_size

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. "
                f"`down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. "
                f"`block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. "
                f"`num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."  # noqa: E501
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."  # noqa: E501
            )

        if isinstance(transformer_layers_per_block, list) and reverse_transformer_layers_per_block is None:
            for layer_number_per_block in transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError("Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet.")

        if (
            isinstance(temporal_transformer_layers_per_block, list)
            and reverse_temporal_transformer_layers_per_block is None
        ):
            for layer_number_per_block in temporal_transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError(
                        "Must provide 'reverse_temporal_transformer_layers_per_block` if using asymmetrical motion module in UNet."
                    )

        # input
        conv_in_kernel = 3
        conv_out_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=conv_in_kernel,
            pad_mode="pad",
            padding=conv_in_padding,
            has_bias=True,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, 0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn=act_fn, cond_proj_dim=time_cond_proj_dim
        )

        if encoder_hid_dim_type is None:
            self.encoder_hid_proj = None

        if addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, True, 0)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        # class embedding
        down_blocks = []
        up_blocks = []

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if isinstance(reverse_transformer_layers_per_block, int):
            reverse_transformer_layers_per_block = [reverse_transformer_layers_per_block] * len(down_block_types)

        if isinstance(temporal_transformer_layers_per_block, int):
            temporal_transformer_layers_per_block = [temporal_transformer_layers_per_block] * len(down_block_types)

        if isinstance(reverse_temporal_transformer_layers_per_block, int):
            reverse_temporal_transformer_layers_per_block = [reverse_temporal_transformer_layers_per_block] * len(
                down_block_types
            )

        if isinstance(motion_num_attention_heads, int):
            motion_num_attention_heads = (motion_num_attention_heads,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == "CrossAttnDownBlockMotion":
                down_block = CrossAttnDownBlockMotion(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim[i],
                    downsample_padding=downsample_padding,
                    add_downsample=not is_final_block,
                    use_linear_projection=use_linear_projection,
                    temporal_num_attention_heads=motion_num_attention_heads[i],
                    temporal_max_seq_length=motion_max_seq_length,
                    temporal_transformer_layers_per_block=temporal_transformer_layers_per_block[i],
                )
            elif down_block_type == "DownBlockMotion":
                down_block = DownBlockMotion(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block[i],
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_downsample=not is_final_block,
                    downsample_padding=downsample_padding,
                    temporal_num_attention_heads=motion_num_attention_heads[i],
                    temporal_max_seq_length=motion_max_seq_length,
                    temporal_transformer_layers_per_block=temporal_transformer_layers_per_block[i],
                )
            else:
                raise ValueError(
                    "Invalid `down_block_type` encountered. Must be one of `CrossAttnDownBlockMotion` or `DownBlockMotion`"
                )

            down_blocks.append(down_block)
        self.down_blocks = nn.CellList(down_blocks)

        # mid: only definition, binding attribute to UNetMotionModel later to maintain the order of sub-modules within
        # UNetMotionModel as self.down_blocks -> self.up_blocks -> self.mid_block, ensuring the correct sequence of
        # sub-modules is loaded when the ip-adpater is loaded.
        if transformer_layers_per_mid_block is None:
            transformer_layers_per_mid_block = (
                transformer_layers_per_block[-1] if isinstance(transformer_layers_per_block[-1], int) else 1
            )

        if use_motion_mid_block:
            mid_block = UNetMidBlockCrossAttnMotion(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=False,
                use_linear_projection=use_linear_projection,
                num_layers=mid_block_layers,
                temporal_num_attention_heads=motion_num_attention_heads[-1],
                temporal_max_seq_length=motion_max_seq_length,
                transformer_layers_per_block=transformer_layers_per_mid_block,
                temporal_transformer_layers_per_block=temporal_transformer_layers_per_mid_block,
            )

        else:
            mid_block = UNetMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=False,
                use_linear_projection=use_linear_projection,
                num_layers=mid_block_layers,
                transformer_layers_per_block=transformer_layers_per_mid_block,
            )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        layers_per_resnet_in_up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_motion_num_attention_heads = list(reversed(motion_num_attention_heads))

        if reverse_transformer_layers_per_block is None:
            reverse_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        if reverse_temporal_transformer_layers_per_block is None:
            reverse_temporal_transformer_layers_per_block = list(reversed(temporal_transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            if up_block_type == "CrossAttnUpBlockMotion":
                up_block = CrossAttnUpBlockMotion(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    resolution_idx=i,
                    num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=reverse_transformer_layers_per_block[i],
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    num_attention_heads=reversed_num_attention_heads[i],
                    cross_attention_dim=reversed_cross_attention_dim[i],
                    add_upsample=add_upsample,
                    use_linear_projection=use_linear_projection,
                    temporal_num_attention_heads=reversed_motion_num_attention_heads[i],
                    temporal_max_seq_length=motion_max_seq_length,
                    temporal_transformer_layers_per_block=reverse_temporal_transformer_layers_per_block[i],
                )
            elif up_block_type == "UpBlockMotion":
                up_block = UpBlockMotion(
                    in_channels=input_channel,
                    prev_output_channel=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    resolution_idx=i,
                    num_layers=reversed_layers_per_block[i] + 1,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_upsample=add_upsample,
                    temporal_num_attention_heads=reversed_motion_num_attention_heads[i],
                    temporal_max_seq_length=motion_max_seq_length,
                    temporal_transformer_layers_per_block=reverse_temporal_transformer_layers_per_block[i],
                )
            else:
                raise ValueError(
                    "Invalid `up_block_type` encountered. Must be one of `CrossAttnUpBlockMotion` or `UpBlockMotion`"
                )

            up_blocks.append(up_block)
            prev_output_channel = output_channel
            layers_per_resnet_in_up_blocks.append(len(up_block.resnets))
        self.up_blocks = nn.CellList(up_blocks)
        self.layers_per_resnet_in_up_blocks = layers_per_resnet_in_up_blocks

        # bind mid_block to self here
        self.mid_block = mid_block

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
            self.conv_act = nn.SiLU()
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=conv_out_kernel,
            pad_mode="pad",
            padding=conv_out_padding,
            has_bias=True,
        )

    @classmethod
    def from_unet2d(
        cls,
        unet: UNet2DConditionModel,
        motion_adapter: Optional[MotionAdapter] = None,
        load_weights: bool = True,
    ):
        has_motion_adapter = motion_adapter is not None

        if has_motion_adapter:
            # check compatibility of number of blocks
            if len(unet.config["down_block_types"]) != len(motion_adapter.config["block_out_channels"]):
                raise ValueError("Incompatible Motion Adapter, got different number of blocks")

            # check layers compatibility for each block
            if isinstance(unet.config["layers_per_block"], int):
                expanded_layers_per_block = [unet.config["layers_per_block"]] * len(unet.config["down_block_types"])
            else:
                expanded_layers_per_block = list(unet.config["layers_per_block"])
            if isinstance(motion_adapter.config["motion_layers_per_block"], int):
                expanded_adapter_layers_per_block = [motion_adapter.config["motion_layers_per_block"]] * len(
                    motion_adapter.config["block_out_channels"]
                )
            else:
                expanded_adapter_layers_per_block = list(motion_adapter.config["motion_layers_per_block"])
            if expanded_layers_per_block != expanded_adapter_layers_per_block:
                raise ValueError("Incompatible Motion Adapter, got different number of layers per block")

        # based on https://github.com/guoyww/AnimateDiff/blob/895f3220c06318ea0760131ec70408b466c49333/animatediff/models/unet.py#L459
        config = dict(unet.config)
        config["_class_name"] = cls.__name__

        down_blocks = []
        for down_blocks_type in config["down_block_types"]:
            if "CrossAttn" in down_blocks_type:
                down_blocks.append("CrossAttnDownBlockMotion")
            else:
                down_blocks.append("DownBlockMotion")
        config["down_block_types"] = down_blocks

        up_blocks = []
        for down_blocks_type in config["up_block_types"]:
            if "CrossAttn" in down_blocks_type:
                up_blocks.append("CrossAttnUpBlockMotion")
            else:
                up_blocks.append("UpBlockMotion")
        config["up_block_types"] = up_blocks

        if has_motion_adapter:
            config["motion_num_attention_heads"] = motion_adapter.config["motion_num_attention_heads"]
            config["motion_max_seq_length"] = motion_adapter.config["motion_max_seq_length"]
            config["use_motion_mid_block"] = motion_adapter.config["use_motion_mid_block"]
            config["layers_per_block"] = motion_adapter.config["motion_layers_per_block"]
            config["temporal_transformer_layers_per_mid_block"] = motion_adapter.config[
                "motion_transformer_layers_per_mid_block"
            ]
            config["temporal_transformer_layers_per_block"] = motion_adapter.config[
                "motion_transformer_layers_per_block"
            ]
            config["motion_num_attention_heads"] = motion_adapter.config["motion_num_attention_heads"]

            # For PIA UNets we need to set the number input channels to 9
            if motion_adapter.config["conv_in_channels"]:
                config["in_channels"] = motion_adapter.config["conv_in_channels"]

        # Need this for backwards compatibility with UNet2DConditionModel checkpoints
        if not config.get("num_attention_heads"):
            config["num_attention_heads"] = config["attention_head_dim"]

        expected_kwargs, optional_kwargs = cls._get_signature_keys(cls)
        config = FrozenDict({k: config.get(k) for k in config if k in expected_kwargs or k in optional_kwargs})
        config["_class_name"] = cls.__name__
        model = cls.from_config(config)

        # Move dtype conversion code here to avoid dtype mismatch issues when loading weights
        # ensure that the Motion UNet is the same dtype as the UNet2DConditionModel
        model.to(unet.dtype)

        if not load_weights:
            return model

        # Logic for loading PIA UNets which allow the first 4 channels to be any UNet2DConditionModel conv_in weight
        # while the last 5 channels must be PIA conv_in weights.
        if has_motion_adapter and motion_adapter.config["conv_in_channels"]:
            model.conv_in = motion_adapter.conv_in
            updated_conv_in_weight = ops.cat([unet.conv_in.weight, motion_adapter.conv_in.weight[:, 4:, :, :]], axis=1)
            ms.load_param_into_net(model.conv_in, {"weight": updated_conv_in_weight, "bias": unet.conv_in.bias})
        else:
            ms.load_param_into_net(model.conv_in, unet.conv_in.parameters_dict())

        ms.load_param_into_net(model.time_proj, unet.time_proj.parameters_dict())
        ms.load_param_into_net(model.time_embedding, unet.time_embedding.parameters_dict())

        if any(isinstance(proc, IPAdapterAttnProcessor) for proc in unet.attn_processors.values()):
            attn_procs = {}
            for name, processor in unet.attn_processors.items():
                if name.endswith("attn1.processor"):
                    attn_processor_class = AttnProcessor
                    attn_procs[name] = attn_processor_class()
                else:
                    attn_processor_class = IPAdapterAttnProcessor
                    attn_procs[name] = attn_processor_class(
                        hidden_size=processor.hidden_size,
                        cross_attention_dim=processor.cross_attention_dim,
                        scale=processor.scale,
                        num_tokens=processor.num_tokens,
                    )
            for name, processor in model.attn_processors.items():
                if name not in attn_procs:
                    attn_procs[name] = processor.__class__()
            model.set_attn_processor(attn_procs)
            model.config.encoder_hid_dim_type = "ip_image_proj"
            model.encoder_hid_proj = unet.encoder_hid_proj

        for i, down_block in enumerate(unet.down_blocks):
            ms.load_param_into_net(model.down_blocks[i].resnets, down_block.resnets.parameters_dict())
            if hasattr(model.down_blocks[i], "attentions"):
                ms.load_param_into_net(model.down_blocks[i].attentions, down_block.attentions.parameters_dict())
            if model.down_blocks[i].downsamplers:
                ms.load_param_into_net(model.down_blocks[i].downsamplers, down_block.downsamplers.parameters_dict())

        for i, up_block in enumerate(unet.up_blocks):
            ms.load_param_into_net(model.up_blocks[i].resnets, up_block.resnets.parameters_dict())
            if hasattr(model.up_blocks[i], "attentions"):
                ms.load_param_into_net(model.up_blocks[i].attentions, up_block.attentions.parameters_dict())
            if model.up_blocks[i].upsamplers:
                ms.load_param_into_net(model.up_blocks[i].upsamplers, up_block.upsamplers.parameters_dict())

        ms.load_param_into_net(model.mid_block.resnets, unet.mid_block.resnets.parameters_dict())
        ms.load_param_into_net(model.mid_block.attentions, unet.mid_block.attentions.parameters_dict())

        if unet.conv_norm_out is not None:
            ms.load_param_into_net(model.conv_norm_out, unet.conv_norm_out.parameters_dict())
        if unet.conv_act is not None:
            ms.load_param_into_net(model.conv_act, unet.conv_act.parameters_dict())
        ms.load_param_into_net(model.conv_out, unet.conv_out.parameters_dict())

        if has_motion_adapter:
            model.load_motion_modules(motion_adapter)

        return model

    def freeze_unet2d_params(self) -> None:
        """Freeze the weights of just the UNet2DConditionModel, and leave the motion modules
        unfrozen for fine tuning.
        """
        # Freeze everything
        for param in self.get_parameters():
            param.requires_grad = False

        # Unfreeze Motion Modules
        for down_block in self.down_blocks:
            motion_modules = down_block.motion_modules
            for param in motion_modules.get_parameters():
                param.requires_grad = True

        for up_block in self.up_blocks:
            motion_modules = up_block.motion_modules
            for param in motion_modules.get_parameters():
                param.requires_grad = True

        if hasattr(self.mid_block, "motion_modules"):
            motion_modules = self.mid_block.motion_modules
            for param in motion_modules.get_parameters():
                param.requires_grad = True

    def load_motion_modules(self, motion_adapter: Optional[MotionAdapter]) -> None:
        for i, down_block in enumerate(motion_adapter.down_blocks):
            ms.load_param_into_net(self.down_blocks[i].motion_modules, down_block.motion_modules.parameters_dict())
        for i, up_block in enumerate(motion_adapter.up_blocks):
            ms.load_param_into_net(self.up_blocks[i].motion_modules, up_block.motion_modules.parameters_dict())

        # to support older motion modules that don't have a mid_block
        if hasattr(self.mid_block, "motion_modules"):
            ms.load_param_into_net(
                self.mid_block.motion_modules, motion_adapter.mid_block.motion_modules.parameters_dict()
            )

    def save_motion_modules(
        self,
        save_directory: str,
        is_main_process: bool = True,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> None:
        state_dict = self.parameters_dict()

        # Extract all motion modules
        motion_state_dict = {}
        for k, v in state_dict.items():
            if "motion_modules" in k:
                motion_state_dict[k] = v

        adapter = MotionAdapter(
            block_out_channels=self.config["block_out_channels"],
            motion_layers_per_block=self.config["layers_per_block"],
            motion_norm_num_groups=self.config["norm_num_groups"],
            motion_num_attention_heads=self.config["motion_num_attention_heads"],
            motion_max_seq_length=self.config["motion_max_seq_length"],
            use_motion_mid_block=self.config["use_motion_mid_block"],
        )
        ms.load_param_into_net(adapter, motion_state_dict)
        adapter.save_pretrained(
            save_directory=save_directory,
            is_main_process=is_main_process,
            safe_serialization=safe_serialization,
            variant=variant,
            push_to_hub=push_to_hub,
            **kwargs,
        )

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:  # type: ignore
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: nn.Cell, processors: Dict[str, AttentionProcessor]):  # type: ignore
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.name_cells().items():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.name_cells().items():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):  # type: ignore
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

    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(module: nn.Cell, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.name_cells().values():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.name_cells().values():
            fn_recursive_feed_forward(module, None, 0)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (CrossAttnDownBlockMotion, DownBlockMotion, CrossAttnUpBlockMotion, UpBlockMotion)):
            module.gradient_checkpointing = value

    def construct(
        self,
        sample: ms.Tensor,
        timestep: Union[ms.Tensor, float, int],
        encoder_hidden_states: ms.Tensor,
        timestep_cond: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, ms.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[ms.Tensor]] = None,
        mid_block_additional_residual: Optional[ms.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[UNetMotionOutput, Tuple[ms.Tensor]]:
        r"""
        The [`UNetMotionModel`] forward method.

        Args:
            sample (`ms.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width`.
            timestep (`ms.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`ms.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            timestep_cond: (`ms.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`ms.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            down_block_additional_residuals: (`tuple` of `ms.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`ms.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_motion_model.UNetMotionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_motion_model.UNetMotionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_motion_model.UNetMotionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if sample.shape[-2] % default_overall_up_factor != 0 or sample.shape[-1] % default_overall_up_factor != 0:
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not ops.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            if isinstance(timestep, float):
                dtype = ms.float64
            else:
                dtype = ms.int64
            timesteps = ms.Tensor([timesteps], dtype=dtype)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None]

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        num_frames = sample.shape[2]
        timesteps = timesteps.broadcast_to((sample.shape[0],))

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.config["addition_embed_type"] == "text_time":
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"  # noqa: E501
                )

            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"  # noqa: E501
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten()).to(text_embeds.dtype)
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = ops.concat([text_embeds, time_embeds], axis=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)

        emb = emb if aug_emb is None else emb + aug_emb
        emb = emb.repeat_interleave(repeats=num_frames, dim=0)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_frames, dim=0)

        if self.encoder_hid_proj is not None and self.config["encoder_hid_dim_type"] == "ip_image_proj":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to "
                    f"'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            image_embeds = self.encoder_hid_proj(image_embeds)
            image_embeds = [image_embed.repeat_interleave(repeats=num_frames, dim=0) for image_embed in image_embeds]
            encoder_hidden_states = (encoder_hidden_states, image_embeds)

        # 2. pre-process
        sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] * num_frames, -1) + sample.shape[3:])
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            # To support older versions of motion modules that don't have a mid_block
            if self.mid_block.has_motion_modules:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-self.layers_per_resnet_in_up_blocks[i] :]
            down_block_res_samples = down_block_res_samples[: -self.layers_per_resnet_in_up_blocks[i]]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)

        sample = self.conv_out(sample)

        # reshape to (batch, channel, framerate, width, height)
        sample = sample[None, :].reshape((-1, num_frames) + sample.shape[1:]).permute(0, 2, 1, 3, 4)

        if not return_dict:
            return (sample,)

        return UNetMotionOutput(sample=sample)
