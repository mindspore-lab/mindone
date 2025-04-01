from typing import Optional, Tuple, Union

import numpy as np
from opensora.models.vae.utils import ChannelChunkConv3d, get_activation, get_conv3d_n_chunks

import mindspore as ms
from mindspore import mint, nn

from mindone.diffusers.models.attention_processor import Attention
from mindone.diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

INTERPOLATE_NUMEL_LIMIT = 2**31 - 1


def chunk_nearest_interpolate(
    x: ms.tensor,
    scale_factor,
):
    limit = INTERPOLATE_NUMEL_LIMIT // np.prod(scale_factor)
    n_chunks = get_conv3d_n_chunks(x.numel(), x.shape[1], limit)
    x_chunks = x.chunk(n_chunks, dim=1)
    x_chunks = [
        mint.nn.functional.interpolate(x_chunk, scale_factor=scale_factor, mode="nearest") for x_chunk in x_chunks
    ]
    return mint.cat(x_chunks, dim=1)


def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, batch_size: int = None):
    seq_len = n_frame * n_hw
    mask = mint.full((seq_len, seq_len), float("-inf"), dtype=dtype)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand((batch_size, -1, -1))
    return mask


class MSReplicationPad5D(nn.Cell):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def construct(self, input_tensor):
        """
        Pads the last three dimensions of a 5D tensor using replicate padding.
        self.padding (Tuple[int, int, int, int, int, int]): The padding size in the form (pad_left, pad_right, pad_up, pad_down, pad_front, pad_back).

        Args:
            input_tensor (Tensor): The input tensor of shape (N, C, D, H, W).

        Returns:
            Tensor: The padded tensor.
        """
        pad_left, pad_right, pad_up, pad_down, pad_front, pad_back = self.padding

        # Pad width (W)
        if pad_left > 0:
            left_pad = mint.repeat_interleave(input_tensor[:, :, :, :, :1], repeats=pad_left, dim=4)
            padded_width = mint.cat((left_pad, input_tensor), dim=4)
        else:
            padded_width = input_tensor

        if pad_right > 0:
            right_pad = mint.repeat_interleave(input_tensor[:, :, :, :, -1:], repeats=pad_right, dim=4)
            padded_width = mint.cat((padded_width, right_pad), dim=4)

        # Pad height (H)
        if pad_up > 0:
            up_pad = mint.repeat_interleave(padded_width[:, :, :, :1, :], repeats=pad_up, dim=3)
            padded_height = mint.cat((up_pad, padded_width), dim=3)
        else:
            padded_height = padded_width

        if pad_down > 0:
            down_pad = mint.repeat_interleave(padded_width[:, :, :, -1:, :], repeats=pad_down, dim=3)
            padded_height = mint.cat((padded_height, down_pad), dim=3)

        # Pad depth (D)
        if pad_front > 0:
            front_pad = mint.repeat_interleave(padded_height[:, :, :1, :, :], repeats=pad_front, dim=2)
            padded_depth = mint.cat((front_pad, padded_height), dim=2)
        else:
            padded_depth = padded_height

        if pad_back > 0:
            back_pad = mint.repeat_interleave(padded_height[:, :, -1:, :, :], repeats=pad_back, dim=2)
            padded_depth = mint.cat((padded_depth, back_pad), dim=2)

        return padded_depth


class CausalConv3d(nn.Cell):
    """
    Implements a causal 3D convolution layer where each position only depends on previous timesteps and current spatial locations.
    This maintains temporal causality in video generation tasks.
    """

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode="replicate",
        **kwargs,
    ):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size - 1,
            0,
        )  # W, H, T
        self.time_causal_padding = padding

        self.conv = ChannelChunkConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        assert self.pad_mode == "replicate", f"pad mode {self.pad_mode} is not supported other than `replicate`"
        self.pad = MSReplicationPad5D(self.time_causal_padding)

    def construct(self, x):
        x = self.pad(x)
        return self.conv(x)


class UpsampleCausal3D(nn.Cell):
    """
    A 3D upsampling layer with an optional convolution.
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        bias=True,
        upsample_factor=(2, 2, 2),
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.upsample_factor = tuple(float(uf) for uf in upsample_factor)  # upsample_factor must be float in MindSpore
        self.conv = CausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, bias=bias)

    def construct(
        self,
        input_tensor: ms.tensor,
    ) -> ms.tensor:
        assert input_tensor.shape[1] == self.channels

        #######################
        # handle hidden states
        #######################
        hidden_states = input_tensor
        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # dtype = hidden_states.dtype
        # if dtype == ms.bfloat16:
        #     hidden_states = hidden_states.to(ms.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # interpolate H & W only for the first frame; interpolate T & H & W for the rest
        T = hidden_states.shape[2]
        first_h, other_h = hidden_states.split((1, T - 1), dim=2)
        # process non-1st frames
        if T > 1:
            other_h = chunk_nearest_interpolate(other_h, scale_factor=self.upsample_factor)
        # proess 1st fram
        first_h = first_h.squeeze(2)
        first_h = chunk_nearest_interpolate(first_h, scale_factor=self.upsample_factor[1:])
        first_h = first_h.unsqueeze(2)
        # concat together
        if T > 1:
            hidden_states = mint.cat((first_h, other_h), dim=2)
        else:
            hidden_states = first_h

        # If the input is bfloat16, we cast back to bfloat16
        # if dtype == ms.bfloat16:
        #     hidden_states = hidden_states.to(dtype)

        hidden_states = self.conv(hidden_states)

        return hidden_states


class DownsampleCausal3D(nn.Cell):
    """
    A 3D downsampling layer with an optional convolution.
    """

    def __init__(
        self,
        channels: int,
        kernel_size=3,
        bias=True,
        stride=2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.conv = CausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, bias=bias)

    def construct(self, input_tensor: ms.tensor) -> ms.tensor:
        assert input_tensor.shape[1] == self.channels
        hidden_states = self.conv(input_tensor)

        return hidden_states


class ResnetBlockCausal3D(nn.Cell):
    r"""
    A Resnet block.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
        conv_3d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = mint.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1)
        self.norm2 = mint.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = mint.nn.Dropout(dropout)
        conv_3d_out_channels = conv_3d_out_channels or out_channels
        self.conv2 = CausalConv3d(out_channels, conv_3d_out_channels, kernel_size=3, stride=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None

        self.use_in_shortcut = self.in_channels != conv_3d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(
                in_channels,
                conv_3d_out_channels,
                kernel_size=1,
                stride=1,
                bias=conv_shortcut_bias,
            )

    def construct(
        self,
        input_tensor: ms.tensor,
    ) -> ms.tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class UNetMidBlockCausal3D(nn.Cell):
    """
    A 3D UNet mid-block [`UNetMidBlockCausal3D`] with multiple residual blocks and optional attention blocks.
    """

    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = resnet_groups

        # there is always at least one resnet
        resnets = [
            ResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.CellList(attentions)
        self.resnets = nn.CellList(resnets)

    def construct(self, hidden_states: ms.tensor, attention_mask: Optional[ms.tensor]) -> ms.tensor:
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                B, C, T, H, W = hidden_states.shape
                # b c f h w -> b (f h w) c
                hidden_states = mint.permute(hidden_states, (0, 2, 3, 4, 1))
                hidden_states = hidden_states.reshape((hidden_states.shape[0], -1, hidden_states.shape[-1]))
                hidden_states = attn(hidden_states, attention_mask=attention_mask)
                # b (f h w) c -> b c f h w
                hidden_states = mint.permute(hidden_states, (0, 2, 1))
                hidden_states = hidden_states.reshape((hidden_states.shape[0], hidden_states.shape[1], T, H, W))
            hidden_states = resnet(hidden_states)

        return hidden_states


class DownEncoderBlockCausal3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_stride: int = 2,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.CellList(resnets)

        if add_downsample:
            self.downsamplers = nn.CellList(
                [
                    DownsampleCausal3D(
                        out_channels,
                        stride=downsample_stride,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def construct(self, hidden_states: ms.tensor) -> ms.tensor:
        for resnet in self.resnets:
            # hidden_states = auto_grad_checkpoint(resnet, hidden_states)  # TODO: check
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                # hidden_states = auto_grad_checkpoint(downsampler, hidden_states)
                hidden_states = downsampler(hidden_states)

        return hidden_states


class UpDecoderBlockCausal3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        upsample_scale_factor=(2, 2, 2),
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.CellList(resnets)

        if add_upsample:
            self.upsamplers = nn.CellList(
                [
                    UpsampleCausal3D(
                        out_channels,
                        out_channels=out_channels,
                        upsample_factor=upsample_scale_factor,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

    def construct(self, hidden_states: ms.tensor) -> ms.tensor:
        for resnet in self.resnets:
            # hidden_states = auto_grad_checkpoint(resnet, hidden_states)
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                # hidden_states = auto_grad_checkpoint(upsampler, hidden_states)
                hidden_states = upsampler(hidden_states)

        return hidden_states
