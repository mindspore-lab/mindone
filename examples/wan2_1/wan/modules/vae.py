# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
from typing import List, Optional, Tuple, Union

import numpy as np

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.nn.utils import no_init_parameters

from mindone.models.utils import zeros_

from ..utils.utils import load_pth

__all__ = ["WanVAE"]

CACHE_T = 2


class CausalConv3d(mint.nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def construct(self, x: Tensor, cache_x: Optional[Tensor] = None) -> Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = mint.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().construct(x)


class RMS_norm(nn.Cell):
    def __init__(
        self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False, dtype: ms.Type = ms.float32
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = Parameter(Tensor(np.ones(shape), dtype=dtype))
        self.bias = Parameter(Tensor(np.zeros(shape), dtype=dtype)) if bias else 0.0

    def construct(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(mint.nn.Upsample):
    def construct(self, x: Tensor) -> Tensor:
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().construct(x.float()).type_as(x)


class Resample(nn.Cell):
    def __init__(self, dim: int, mode: str, dtype: ms.Type = ms.float32) -> None:
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.SequentialCell(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest"),
                mint.nn.Conv2d(dim, dim // 2, 3, padding=1, dtype=dtype),
            )
        elif mode == "upsample3d":
            self.resample = nn.SequentialCell(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest"),
                mint.nn.Conv2d(dim, dim // 2, 3, padding=1, dtype=dtype),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0), dtype=dtype)

        elif mode == "downsample2d":
            self.resample = nn.SequentialCell(
                mint.nn.ZeroPad2d((0, 1, 0, 1)), mint.nn.Conv2d(dim, dim, 3, stride=(2, 2), dtype=dtype)
            )
        elif mode == "downsample3d":
            self.resample = nn.SequentialCell(
                nn.ZeroPad2d((0, 1, 0, 1)), mint.nn.Conv2d(dim, dim, 3, stride=(2, 2), dtype=dtype)
            )
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), dtype=dtype)

        else:
            self.resample = mint.nn.Identity()

    def construct(self, x: Tensor, feat_cache: Optional[Tensor] = None, feat_idx: List[int] = [0]) -> Tensor:
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
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, dtype: ms.Type = ms.float32) -> None:
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

    def construct(self, x: Tensor, feat_cache: Optional[Tensor] = None, feat_idx: List[int] = [0]) -> Tensor:
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

    def __init__(self, dim: int, dtype: ms.Type = ms.float32) -> None:
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim, dtype=dtype)
        self.to_qkv = mint.nn.Conv2d(dim, dim * 3, 1, dtype=dtype)
        self.proj = mint.nn.Conv2d(dim, dim, 1, dtype=dtype)

        # zero out the last layer params
        zeros_(self.proj.weight)

    def construct(self, x: Tensor) -> Tensor:
        identity = x
        b, c, t, h, w = x.shape
        x = x.transpose(1, 2).reshape(-1, c, h, w)  # b c t h w -> (b t) c h w
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = ops.flash_attention_score(q, k, v, 1, scalar_value=1 / math.sqrt(q.shape[-1]), input_layout="BNSD")
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = x.reshape(b, t, *x.shape[1:]).transpose(1, 2)  # (b t) c h w-> b c t h w
        return x + identity


class Encoder3d(nn.Cell):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [True, True, False],
        dropout: float = 0.0,
        dtype: ms.Type = ms.float32,
    ) -> None:
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
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1, dtype=dtype)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout, dtype=dtype))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim, dtype=dtype))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode, dtype=dtype))
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

    def construct(self, x: Tensor, feat_cache: Optional[Tensor] = None, feat_idx: List[int] = [0]) -> Tensor:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
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
                    # cache last frame of last two chunk
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
        attn_scales: List[float] = [],
        temperal_upsample: List[bool] = [False, True, True],
        dropout=0.0,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

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
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout, dtype=dtype))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim, dtype=dtype))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode, dtype=dtype))
                scale *= 2.0
        self.upsamples = nn.SequentialCell(*upsamples)

        # output blocks
        self.head = nn.SequentialCell(
            RMS_norm(out_dim, images=False, dtype=dtype),
            mint.nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1, dtype=dtype),
        )

    def construct(self, x: Tensor, feat_cache: Optional[Tensor] = None, feat_idx: List[int] = [0]) -> Tensor:
        # conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = mint.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
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
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [True, True, False],
        dropout: float = 0.0,
        dtype: ms.Type = ms.float32,
    ) -> None:
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
            dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout, dtype=dtype
        )

    def construct(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x: Tensor, scale: List[Union[float, Tensor]]) -> Tensor:
        self.clear_cache()
        # cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        # 对encode输入的x，按时间拆分为1、4、4、4....
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
        if isinstance(scale[0], Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z: Tensor, scale: List[Union[float, Tensor]]) -> Tensor:
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = mint.cat([out, out_], 2)
        self.clear_cache()
        return out

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = mint.exp(0.5 * log_var)
        eps = mint.randn_like(std)
        return eps * std + mu

    def sample(self, imgs: Tensor, deterministic: bool = False) -> Tensor:
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


def _video_vae(pretrained_path: Optional[str] = None, z_dim: Optional[int] = None, **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)

    # init model
    with no_init_parameters():
        model = WanVAE_(**cfg)

    # load checkpoint
    if pretrained_path is not None:
        logging.info(f"loading {pretrained_path}")
        if pretrained_path.endswith(".pth"):
            param_dict = load_pth(pretrained_path, dtype=model.dtype)
            ms.load_param_into_net(model, param_dict)
        else:
            ms.load_checkpoint(pretrained_path, model)
    model.init_parameters_data()
    return model


class WanVAE:
    def __init__(self, z_dim: int = 16, vae_pth: Optional[str] = None, dtype=ms.float32) -> None:
        self.dtype = dtype

        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = Tensor(mean, dtype=dtype)
        self.std = Tensor(std, dtype=dtype)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(pretrained_path=vae_pth, z_dim=z_dim, dtype=dtype)
        self.model.set_train(False)
        for param in self.model.trainable_params():
            param.requires_grad = False

    def encode(self, videos: List[Tensor]) -> List[Tensor]:
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        return [self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0) for u in videos]

    def decode(self, zs: List[Tensor]) -> List[Tensor]:
        return [self.model.decode(u.unsqueeze(0), self.scale).float().clamp_(-1, 1).squeeze(0) for u in zs]
