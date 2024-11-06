"""FIXME: a patch to support dynamic shape training. Drop this module once the official models supports dynanmic shape"""
from typing import Optional, Tuple

import mindspore.nn as nn

from mindone.diffusers import AutoencoderKL as AutoencoderKL_
from mindone.diffusers.configuration_utils import register_to_config

from .vae import Encoder


class AutoencoderKL(AutoencoderKL_):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        force_upcast: float = True,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            force_upcast=force_upcast,
            latents_mean=latents_mean,
            latents_std=latents_std,
            use_quant_conv=use_quant_conv,
            use_post_quant_conv=use_post_quant_conv,
        )

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        self.quant_conv = (
            nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1, pad_mode="pad", has_bias=True)
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            nn.Conv2d(latent_channels, latent_channels, 1, pad_mode="pad", has_bias=True)
            if use_post_quant_conv
            else None
        )
