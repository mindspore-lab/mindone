"""FIXME: a patch to support dynamic shape training. Drop this module once the official models supports dynanmic shape"""
from typing import Optional

import mindspore.nn as nn

from mindone.diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D as DownEncoderBlock2D_
from mindone.diffusers.models.unets.unet_2d_blocks import ResnetBlockCondNorm2D
from mindone.diffusers.models.unets.unet_2d_blocks import get_down_block as get_down_block_

from ..resnet import ResnetBlock2D


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
):
    if down_block_type == "DownEncoderBlock2D":
        return DownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    return get_down_block_(
        down_block_type=down_block_type,
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        add_downsample=add_downsample,
        resnet_eps=resnet_eps,
        resnet_act_fn=resnet_act_fn,
        transformer_layers_per_block=transformer_layers_per_block,
        num_attention_heads=num_attention_heads,
        resnet_groups=resnet_groups,
        cross_attention_dim=cross_attention_dim,
        downsample_padding=downsample_padding,
        dual_cross_attention=dual_cross_attention,
        use_linear_projection=use_linear_projection,
        only_cross_attention=only_cross_attention,
        upcast_attention=upcast_attention,
        resnet_time_scale_shift=resnet_time_scale_shift,
        attention_type=attention_type,
        resnet_skip_time_act=resnet_skip_time_act,
        resnet_out_scale_factor=resnet_out_scale_factor,
        cross_attention_norm=cross_attention_norm,
        attention_head_dim=attention_head_dim,
        downsample_type=downsample_type,
        dropout=dropout,
    )


class DownEncoderBlock2D(DownEncoderBlock2D_):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_pre_norm=resnet_pre_norm,
            output_scale_factor=output_scale_factor,
            add_downsample=add_downsample,
            downsample_padding=downsample_padding,
        )

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            if resnet_time_scale_shift == "spatial":
                resnets.append(
                    ResnetBlockCondNorm2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        temb_channels=None,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm="spatial",
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                    )
                )
            else:
                resnets.append(
                    ResnetBlock2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        temb_channels=None,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                    )
                )

        self.resnets = nn.CellList(resnets)
