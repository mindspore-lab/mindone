"""FIXME: a patch to support dynamic shape training. Drop this module once the official models supports dynanmic shape"""
from typing import Optional

import mindspore as ms
import mindspore.nn as nn

from mindone.diffusers.models.resnet import ResnetBlock2D as ResnetBlock2D_


class ResnetBlock2D(ResnetBlock2D_):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        kernel: Optional[ms.Tensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_shortcut=conv_shortcut,
            dropout=dropout,
            temb_channels=temb_channels,
            groups=groups,
            groups_out=groups_out,
            pre_norm=pre_norm,
            eps=eps,
            non_linearity=non_linearity,
            skip_time_act=skip_time_act,
            time_embedding_norm=time_embedding_norm,
            kernel=kernel,
            output_scale_factor=output_scale_factor,
            use_in_shortcut=use_in_shortcut,
            up=up,
            down=down,
            conv_shortcut_bias=conv_shortcut_bias,
            conv_2d_out_channels=conv_2d_out_channels,
        )
        out_channels = in_channels if out_channels is None else out_channels
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                pad_mode="pad",
                has_bias=conv_shortcut_bias,
            )
