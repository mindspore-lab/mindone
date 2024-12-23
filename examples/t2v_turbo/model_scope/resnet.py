from typing import Optional

import mindspore as ms
from mindspore import nn

from mindone.diffusers.models.activations import SiLU
from mindone.diffusers.models.normalization import GroupNorm, get_activation
from mindone.diffusers.models.resnet import AlphaBlender, ResnetBlock2D


class TemporalConvLayer(nn.Cell):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016

    Parameters:
        in_dim (`int`): Number of input channels.
        out_dim (`int`): Number of output channels.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        dtype: ms.dtype = ms.float32,
    ):
        super().__init__()

        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # conv layers
        self.conv1 = nn.SequentialCell(
            GroupNorm(norm_num_groups, in_dim),
            SiLU(),
            nn.Conv3d(
                in_dim, out_dim, (3, 1, 1), padding=(1, 1, 0, 0, 0, 0), pad_mode="pad", has_bias=True, dtype=dtype
            ).to_float(dtype),
        )
        self.conv2 = nn.SequentialCell(
            GroupNorm(norm_num_groups, out_dim),
            SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(
                out_dim, in_dim, (3, 1, 1), padding=(1, 1, 0, 0, 0, 0), pad_mode="pad", has_bias=True, dtype=dtype
            ).to_float(dtype),
        )
        self.conv3 = nn.SequentialCell(
            GroupNorm(norm_num_groups, out_dim),
            SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(
                out_dim, in_dim, (3, 1, 1), padding=(1, 1, 0, 0, 0, 0), pad_mode="pad", has_bias=True, dtype=dtype
            ).to_float(dtype),
        )
        self.conv4 = nn.SequentialCell(
            GroupNorm(norm_num_groups, out_dim),
            SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(
                out_dim,
                in_dim,
                (3, 1, 1),
                padding=(1, 1, 0, 0, 0, 0),
                pad_mode="pad",
                has_bias=True,
                weight_init="zeros",
                bias_init="zeros",
                dtype=dtype,
            ).to_float(
                dtype
            ),  # zero out the last layer params,so the conv block is identity
        )

    def construct(self, hidden_states: ms.Tensor, num_frames: int = 1) -> ms.Tensor:
        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )

        identity = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)

        hidden_states = identity + hidden_states

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )
        return hidden_states


class TemporalResnetBlock(nn.Cell):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        dtype: ms.dtype = ms.float32,
    ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        kernel_size = (3, 1, 1)
        # padding = [k // 2 for k in kernel_size]
        padding = (1, 1, 0, 0, 0, 0)

        self.norm1 = GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            pad_mode="pad",
            padding=padding,
            has_bias=True,
            dtype=dtype,
        ).to_float(dtype)

        if temb_channels is not None:
            self.time_emb_proj = nn.Dense(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = GroupNorm(num_groups=32, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = nn.Dropout(p=0.0)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            pad_mode="pad",
            padding=padding,
            has_bias=True,
            dtype=dtype,
        ).to_float(dtype)

        self.nonlinearity = get_activation("silu")()

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                pad_mode="pad",
                padding=0,
                has_bias=True,
                dtype=dtype,
            ).to_float(dtype)

    def construct(self, input_tensor: ms.Tensor, temb: ms.Tensor) -> ms.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, :, None, None]
            temb = temb.permute(0, 2, 1, 3, 4)
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


# VideoResBlock
class SpatioTemporalResBlock(nn.Cell):
    r"""
    A SpatioTemporal Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the spatial resenet.
        temporal_eps (`float`, *optional*, defaults to `eps`): The epsilon to use for the temporal resnet.
        merge_factor (`float`, *optional*, defaults to `0.5`): The merge factor to use for the temporal mixing.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        temporal_eps: Optional[float] = None,
        merge_factor: float = 0.5,
        merge_strategy="learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
        dtype: ms.dtype = ms.float32,
    ):
        super().__init__()

        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=eps,
        )

        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels if out_channels is not None else in_channels,
            out_channels=out_channels if out_channels is not None else in_channels,
            temb_channels=temb_channels,
            eps=temporal_eps if temporal_eps is not None else eps,
            dtype=dtype,
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
        )

    def construct(
        self,
        hidden_states: ms.Tensor,
        temb: Optional[ms.Tensor] = None,
        image_only_indicator: Optional[ms.Tensor] = None,
    ):
        num_frames = image_only_indicator.shape[-1]
        hidden_states = self.spatial_res_block(hidden_states, temb)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )

        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)

        hidden_states = self.temporal_res_block(hidden_states, temb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        return hidden_states
