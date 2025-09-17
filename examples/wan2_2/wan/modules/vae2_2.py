# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
from typing import Any, List, Optional, Tuple

import numpy as np

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
import mindspore.nn as nn

from mindone.models.utils import zeros_

from ..utils.utils import load_pth

__all__ = ["Wan2_2_VAE"]

CACHE_T = 2


class CausalConv3d(mint.nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, dtype: Any = ms.float32, **kwargs):
        super().__init__(*args, dtype=dtype, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def construct(self, x: ms.Tensor, cache_x: Optional[ms.Tensor] = None) -> ms.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = mint.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().construct(x)


class RMS_norm(nn.Cell):
    def __init__(
        self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False, dtype: Any = ms.float32
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = ms.Parameter(ms.Tensor(np.ones(shape), dtype=dtype))
        self.bias = ms.Parameter(ms.Tensor(np.zeros(shape), dtype=dtype)) if bias else 0.0

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(mint.nn.Upsample):
    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().construct(x.float()).type_as(x)


class Resample(nn.Cell):
    def __init__(self, dim: int, mode: str, dtype: Any = ms.float32):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.SequentialCell(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest"),
                mint.nn.Conv2d(dim, dim, 3, padding=1, dtype=dtype),
            )
        elif mode == "upsample3d":
            self.resample = nn.SequentialCell(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest"),
                mint.nn.Conv2d(dim, dim, 3, padding=1, dtype=dtype),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0), dtype=dtype)
        elif mode == "downsample2d":
            self.resample = nn.SequentialCell(
                mint.nn.ZeroPad2d((0, 1, 0, 1)), mint.nn.Conv2d(dim, dim, 3, stride=(2, 2), dtype=dtype)
            )
        elif mode == "downsample3d":
            self.resample = nn.SequentialCell(
                mint.nn.ZeroPad2d((0, 1, 0, 1)), mint.nn.Conv2d(dim, dim, 3, stride=(2, 2), dtype=dtype)
            )
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), dtype=dtype)
        else:
            self.resample = mint.nn.Identity()

    def construct(
        self,
        x: ms.Tensor,
        feat_cache: Optional[List[Optional[ms.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> ms.Tensor:
        b, c, t, h, w = x.shape
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        # cache last frame of last two chunk
                        cache_x = mint.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = mint.cat([mint.zeros_like(cache_x), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = mint.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.transpose(1, 2).flatten(0, 1)  # b c t h w -> (b t) c h w
        x = self.resample(x)
        x = x.reshape(b, t, *x.shape[1:]).transpose(1, 2)  # (b t) c h w -> b c t h w

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(mint.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class ResidualBlock(nn.Cell):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, dtype: Any = ms.float32):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.SequentialCell(
            RMS_norm(in_dim, images=False, dtype=dtype),
            mint.nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1, dtype=dtype),
            RMS_norm(out_dim, images=False, dtype=dtype),
            mint.nn.SiLU(),
            mint.nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1, dtype=dtype),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1, dtype=dtype) if in_dim != out_dim else mint.nn.Identity()

    def construct(
        self,
        x: ms.Tensor,
        feat_cache: Optional[List[Optional[ms.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> ms.Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = mint.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Cell):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim: int, dtype: Any = ms.float32):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim, dtype=dtype)
        self.to_qkv = mint.nn.Conv2d(dim, dim * 3, 1, dtype=dtype)
        self.proj = mint.nn.Conv2d(dim, dim, 1, dtype=dtype)

        # zero out the last layer params
        zeros_(self.proj.weight)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        identity = x
        b, c, t, h, w = x.shape
        x = x.transpose(1, 2).reshape(-1, c, h, w)  # b c t h w -> (b t) c h w
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        x = mint.softmax(x, dim=-1) @ v
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = x.reshape(b, t, *x.shape[1:]).transpose(1, 2)  # (b t) c h w-> b c t h w
        return x + identity


def patchify(x: ms.Tensor, patch_size: int) -> ms.Tensor:
    if patch_size == 1:
        return x
    if len(x.shape) == 4:
        # b c (h q) (w r) -> b (c r q) h w
        x = x.reshape(*x.shape[:2], x.shape[2] // patch_size, patch_size, x.shape[3] // patch_size, patch_size)
        x = x.transpose(0, 1, 5, 3, 2, 4)
        x = x.reshape(x.shape[0], x.shape[1] * patch_size * patch_size, *x.shape[4:])
    elif len(x.shape) == 5:
        # b c f (h q) (w r) -> b (c r q) f h w
        x = x.reshape(*x.shape[:3], x.shape[3] // patch_size, patch_size, x.shape[4] // patch_size, patch_size)
        x = x.transpose(0, 1, 6, 4, 2, 3, 5)
        x = x.reshape(x.shape[0], x.shape[1] * patch_size * patch_size, *x.shape[4:])
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x: ms.Tensor, patch_size: int) -> ms.Tensor:
    if patch_size == 1:
        return x

    if len(x.shape) == 4:
        # b (c r q) h w -> b c (h q) (w r)
        x = x.reshape(x.shape[0], x.shape[1] // (patch_size * patch_size), patch_size, patch_size, *x.shape[2:])
        x = x.transpose(0, 1, 4, 3, 5, 2)
        x = x.reshape(*x.shape[:2], x.shape[2] * patch_size, x.shape[4] * patch_size)
    elif len(x.shape) == 5:
        # b (c r q) f h w -> b c f (h q) (w r)
        x = x.reshape(x.shape[0], x.shape[1] // (patch_size * patch_size), patch_size, patch_size, *x.shape[2:])
        x = x.transpose(0, 1, 4, 5, 3, 6, 2)
        x = x.reshape(*x.shape[:3], x.shape[3] * patch_size, x.shape[5] * patch_size)
    return x


class AvgDown3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.view(B, self.out_channels, self.group_size, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def construct(self, x: ms.Tensor, first_chunk: bool = False) -> ms.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.shape[0],
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.shape[2],
            x.shape[3],
            x.shape[4],
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.shape[0],
            self.out_channels,
            x.shape[2] * self.factor_t,
            x.shape[4] * self.factor_s,
            x.shape[6] * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class Down_ResidualBlock(nn.Cell):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
        dtype: Any = ms.float32,
    ):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim, out_dim, factor_t=2 if temperal_downsample else 1, factor_s=2 if down_flag else 1
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout, dtype=dtype))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode, dtype=dtype))

        self.downsamples = nn.SequentialCell(*downsamples)

    def construct(
        self, x: ms.Tensor, feat_cache: Optional[List[Optional[ms.Tensor]]] = None, feat_idx: List[int] = [0]
    ) -> ms.Tensor:
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Cell):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        dtype: Any = ms.float32,
    ):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2 if up_flag else 1
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout, dtype=dtype))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode, dtype=dtype))

        self.upsamples = nn.SequentialCell(*upsamples)

    def construct(
        self,
        x: ms.Tensor,
        feat_cache: Optional[List[Optional[ms.Tensor]]] = None,
        feat_idx: List[int] = [0],
        first_chunk: bool = False,
    ) -> ms.Tensor:
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class Encoder3d(nn.Cell):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[Any] = [],
        temperal_downsample: List[bool] = [True, True, False],
        dropout: float = 0.0,
        dtype: Any = ms.float32,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1, dtype=dtype)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                    dtype=dtype,
                )
            )
            scale /= 2.0
        self.downsamples = nn.SequentialCell(*downsamples)

        # middle blocks
        self.middle = nn.SequentialCell(
            ResidualBlock(out_dim, out_dim, dropout, dtype=dtype),
            AttentionBlock(out_dim, dtype=dtype),
            ResidualBlock(out_dim, out_dim, dropout, dtype=dtype),
        )

        # output blocks
        self.head = nn.SequentialCell(
            RMS_norm(out_dim, images=False, dtype=dtype),
            mint.nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1, dtype=dtype),
        )

    def construct(
        self,
        x: ms.Tensor,
        feat_cache: Optional[List[Optional[ms.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> ms.Tensor:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = mint.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = mint.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Cell):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[Any] = [],
        temperal_upsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        dtype: Any = ms.float32,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1, dtype=dtype)

        # middle blocks
        self.middle = nn.SequentialCell(
            ResidualBlock(dims[0], dims[0], dropout, dtype=dtype),
            AttentionBlock(dims[0], dtype=dtype),
            ResidualBlock(dims[0], dims[0], dropout, dtype=dtype),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                    dtype=dtype,
                )
            )
        self.upsamples = nn.SequentialCell(*upsamples)

        # output blocks
        self.head = nn.SequentialCell(
            RMS_norm(out_dim, images=False, dtype=dtype),
            mint.nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1, dtype=dtype),
        )

    def construct(
        self,
        x: ms.Tensor,
        feat_cache: Optional[List[Optional[ms.Tensor]]] = None,
        feat_idx: List[int] = [0],
        first_chunk: bool = False,
    ) -> ms.Tensor:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = mint.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = mint.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model: nn.Cell) -> int:
    count = 0
    for _, m in model.cells_and_names():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Cell):
    def __init__(
        self,
        dim: int = 160,
        dec_dim: int = 256,
        z_dim: int = 16,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[Any] = [],
        temperal_downsample: List[bool] = [True, True, False],
        dropout: float = 0.0,
        dtype: Any = ms.float32,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.dtype = dtype

        # modules
        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout, dtype=dtype
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1, dtype=dtype)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1, dtype=dtype)
        self.decoder = Decoder3d(
            dec_dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout, dtype=dtype
        )

    def construct(self, x: ms.Tensor, scale: List[Any] = [0, 1]) -> Tuple[ms.Tensor, ms.Tensor]:
        mu = self.encode(x, scale)
        x_recon = self.decode(mu, scale)
        return x_recon, mu

    def encode(self, x: ms.Tensor, scale: List[Any]) -> ms.Tensor:
        self.clear_cache()
        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = mint.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], ms.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z: ms.Tensor, scale: List[Any]) -> ms.Tensor:
        self.clear_cache()
        if isinstance(scale[0], ms.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=True
                )
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = mint.cat([out, out_], 2)
        out = unpatchify(out, patch_size=2)
        self.clear_cache()
        return out

    def reparameterize(self, mu: ms.Tensor, log_var: ms.Tensor) -> ms.Tensor:
        std = mint.exp(0.5 * log_var)
        eps = mint.randn_like(std)
        return eps * std + mu

    def sample(self, imgs: ms.Tensor, deterministic: bool = False) -> ms.Tensor:
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = mint.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * mint.randn_like(std)

    def clear_cache(self) -> None:
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(
    pretrained_path: Optional[str] = None, z_dim: int = 16, dim: int = 160, dtype: Any = ms.float32, **kwargs
) -> WanVAE_:
    # params
    cfg = dict(
        dim=dim,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)

    # init model
    with nn.no_init_parameters():
        model = WanVAE_(dtype=dtype, **cfg)

    # load checkpoint
    logging.info(f"loading {pretrained_path}")
    param_dict = load_pth(pretrained_path)
    ms.load_param_into_net(model, param_dict, strict_load=True)

    return model


class Wan2_2_VAE:
    def __init__(
        self,
        z_dim: int = 48,
        c_dim: int = 160,
        vae_pth: Optional[str] = None,
        dim_mult: List[int] = [1, 2, 4, 4],
        temperal_downsample: List[bool] = [False, True, True],
        dtype: ms.dtype = ms.float32,
    ):
        self.dtype = dtype

        mean = ms.tensor(
            [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
            dtype=dtype,
        )
        std = ms.tensor(
            [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ],
            dtype=dtype,
        )
        self.scale = [mean, 1.0 / std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
            dim=c_dim,
            dim_mult=dim_mult,
            temperal_downsample=temperal_downsample,
            dtype=dtype,
        )
        self.model.set_train(False)
        for param in self.model.trainable_params():
            param.requires_grad = False

    def encode(self, videos: List[ms.Tensor]) -> Optional[List[ms.Tensor]]:
        try:
            if not isinstance(videos, list):
                raise TypeError("videos should be a list")
            # with amp.autocast(dtype=self.dtype):
            return [self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0) for u in videos]
        except TypeError as e:
            logging.info(e)
            return None

    def decode(self, zs: List[ms.Tensor]) -> Optional[List[ms.Tensor]]:
        try:
            if not isinstance(zs, list):
                raise TypeError("zs should be a list")
            # with amp.autocast(dtype=self.dtype):
            return [self.model.decode(u.unsqueeze(0), self.scale).float().clamp_(-1, 1).squeeze(0) for u in zs]
        except TypeError as e:
            logging.info(e)
            return None
