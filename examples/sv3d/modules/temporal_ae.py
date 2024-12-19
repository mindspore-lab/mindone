import logging
import sys
from typing import Callable, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

from modules.video_model import TemporalTransformerBlock

from mindspore import Parameter, Tensor, float16, nn, ops

sys.path.append("../../stable_diffusion_xl")  # FIXME: loading modules from the SDXL directory
from sgm.modules.diffusionmodules.model import AttnBlock, Decoder, ResnetBlock
from sgm.modules.diffusionmodules.openaimodel import ResBlock
from sgm.modules.diffusionmodules.util import timestep_embedding


class VideoResBlock(ResnetBlock):
    def __init__(
        self,
        out_channels,
        *args,
        dropout=0.0,
        video_kernel_size=3,
        alpha=0.0,
        merge_strategy: Literal["fixed", "learned"] = "learned",
        **kwargs,
    ):
        super().__init__(out_channels=out_channels, dropout=dropout, *args, **kwargs)
        if video_kernel_size is None:
            video_kernel_size = (3, 1, 1)

        self.time_stack = ResBlock(
            channels=out_channels,
            emb_channels=0,
            dropout=dropout,
            dims=3,
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            skip_t_emb=True,
        )

        self.merge_strategy = merge_strategy
        if self.merge_strategy == "fixed":
            self.mix_factor = Tensor([alpha])
        elif self.merge_strategy == "learned":
            self.mix_factor = Parameter([alpha])
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self):
        if self.merge_strategy == "fixed":
            return self.mix_factor
        elif self.merge_strategy == "learned":
            return ops.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError(f"unknown merge strategy {self.merge_strategy}")

    def construct(self, x, temb, skip_video=False, timesteps=None):
        x = super().construct(x, temb)
        if timesteps is None:
            timesteps = 1

        if not skip_video:
            _, c, h, w = x.shape
            x_spat = x.reshape(-1, timesteps, c, h, w).swapaxes(1, 2)  # (b t) c h w -> b c t h w
            x_temp = self.time_stack(x_spat, temb)

            alpha = self.get_alpha()
            x = alpha * x_temp + (1.0 - alpha) * x_spat

            x = x.swapaxes(1, 2).reshape(-1, c, h, w)  # b c t h w -> (b t) c h w

        return x


class Conv3DLayer(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, video_kernel_size: Union[int, tuple], *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.conv3d = nn.Conv3d(
            out_channels, out_channels, kernel_size=video_kernel_size, pad_mode="same", has_bias=True
        ).to_float(float16)

    def construct(self, x: Tensor, timesteps, skip_video=False) -> Tensor:
        x = super().construct(x)
        if skip_video:
            return x
        if timesteps is None:
            timesteps = 1

        # (b t) c h w -> b c t h w
        x = x.reshape(-1, timesteps, x.shape[1], x.shape[2], x.shape[3]).swapaxes(1, 2)
        x = self.conv3d(x)
        x = x.swapaxes(1, 2).reshape(-1, x.shape[1], x.shape[3], x.shape[4])  # b c t h w -> (b t) c h w

        return x


class VideoBlock(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        attn_type: Literal["vanilla", "flash-attention"] = "vanilla",
        alpha: float = 0,
        merge_strategy: Literal["fixed", "learned"] = "learned",
    ):
        super().__init__()

        self.in_channels = in_channels

        if attn_type.lower() == "flash-attention":
            logging.warning("Flash attention is not yet supported for the Attention Block.")
        self.spat_attn = AttnBlock(in_channels)

        # no context, single headed, as in base class
        self.time_mix_block = TemporalTransformerBlock(
            dim=in_channels,
            n_heads=1,
            d_head=in_channels,
            ff_in=True,
            attn_mode=attn_type,
        )

        time_embed_dim = self.in_channels * 4
        self.video_time_embed = nn.SequentialCell(
            nn.Dense(self.in_channels, time_embed_dim),
            nn.SiLU(),
            nn.Dense(time_embed_dim, self.in_channels),
        )

        self.merge_strategy = merge_strategy
        if self.merge_strategy == "fixed":
            self.mix_factor = Tensor([alpha])
        elif self.merge_strategy == "learned":
            self.mix_factor = Parameter([alpha])
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def construct(self, x: Tensor, timesteps: Tensor, skip_video: bool = False):
        if skip_video:
            return self.spat_attn(x)

        x_in = x
        x = self.spat_attn.attention(x)

        b, c, h, w = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(b, -1, c)  # b c h w -> b (h w) c

        x_mix = x
        num_frames = ops.arange(timesteps).repeat(b // timesteps)
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
        emb = self.video_time_embed(t_emb)  # b, n_channels
        emb = emb[:, None, :]
        x_mix = x_mix + emb

        alpha = self.get_alpha()
        x_mix = self.time_mix_block(x_mix, timesteps=timesteps)
        x = alpha * x + (1.0 - alpha) * x_mix  # alpha merge

        x = x.reshape(b, h, w, c).transpose(0, 3, 1, 2)  # b (h w) c -> b c h w
        x = self.spat_attn.proj_out(x)

        return x_in + x

    def get_alpha(self):
        if self.merge_strategy == "fixed":
            return self.mix_factor
        elif self.merge_strategy == "learned":
            return ops.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError(f"unknown merge strategy {self.merge_strategy}")


class VideoDecoder(Decoder):
    def __init__(
        self,
        *args,
        video_kernel_size: Union[int, tuple] = 3,
        alpha: float = 0.0,
        merge_strategy: Literal["fixed", "learned"] = "learned",
        time_mode: Literal["all", "conv-only", "attn-only"] = "conv-only",
        **kwargs,
    ):
        if not isinstance(video_kernel_size, int):  # kludge around OmegaConf
            video_kernel_size = tuple(video_kernel_size)

        self.video_kernel_size = video_kernel_size
        self.alpha = alpha
        self.merge_strategy = merge_strategy
        self.time_mode = time_mode

        super().__init__(*args, **kwargs)

    def get_last_layer(self, skip_time_mix=False, **kwargs):
        if self.time_mode == "attn-only":
            raise NotImplementedError("time mode attn-only not implemented for video")
        else:
            return self.conv_out.time_mix_conv.weight if not skip_time_mix else self.conv_out.weight

    def _make_attn(self) -> Callable:
        default = super()._make_attn()

        def _make_attn(*args, **kwargs):
            if self.time_mode not in ["conv-only", "only-last-conv"]:
                return VideoBlock(*args, **kwargs, alpha=self.alpha, merge_strategy=self.merge_strategy)
            else:
                return default(*args, **kwargs)

        return _make_attn

    def _make_resblock(self) -> Callable:
        default = super()._make_resblock()

        def _make_resblock(*args, **kwargs):
            if self.time_mode not in ["attn-only", "only-last-conv"]:
                return VideoResBlock(
                    *args,
                    video_kernel_size=self.video_kernel_size,
                    alpha=self.alpha,
                    merge_strategy=self.merge_strategy,
                    **kwargs,
                )
            else:
                return default(*args, **kwargs)

        return _make_resblock

    def _make_conv(self) -> Callable:
        default = super()._make_conv()

        def _make_conv(*args, **kwargs):
            if self.time_mode != "attn-only":
                return Conv3DLayer(*args, **kwargs, video_kernel_size=self.video_kernel_size)
            else:
                return default(*args, **kwargs)

        return _make_conv
