torch_npu = None
npu_config = None

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) or isinstance(t, list) else ((t,) * length)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        enable_cached=False,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = kwargs.pop("stride", 1)
        self.padding = kwargs.pop("padding", 0)
        self.padding = list(cast_tuple(self.padding, 3))
        self.padding[0] = 0
        self.stride = cast_tuple(self.stride, 3)
        self.conv = nn.Conv3d(chan_in, chan_out, self.kernel_size, stride=self.stride, padding=self.padding, bias=bias)
        self.enable_cached = enable_cached
        self.causal_cached = None
        self.cache_offset = 0

    def forward(self, x):
        x_dtype = x.dtype
        if self.causal_cached is None:
            first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))
        else:
            first_frame_pad = self.causal_cached
        x = torch.concatenate((first_frame_pad, x), dim=2)

        if self.enable_cached and self.time_kernel_size != 1:
            if (self.time_kernel_size - 1) // self.stride[0] != 0:
                if self.cache_offset == 0:
                    self.causal_cached = x[:, :, -(self.time_kernel_size - 1) // self.stride[0] :]
                else:
                    self.causal_cached = x[:, :, : -self.cache_offset][
                        :, :, -(self.time_kernel_size - 1) // self.stride[0] :
                    ]
            else:
                self.causal_cached = x[:, :, 0:0, :, :]

        if npu_config is not None and npu_config.on_npu:
            return npu_config.run_conv3d(self.conv, x, x_dtype)
        else:
            x = self.conv(x)
            return x


def video_to_image(func):
    def wrapper(self, x, *args, **kwargs):
        if x.dim() == 5:
            t = x.shape[2]
            if True:
                x = rearrange(x, "b c t h w -> (b t) c h w")
                x = func(self, x, *args, **kwargs)
                x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            else:
                # Conv 2d slice infer
                result = []
                for i in range(t):
                    frame = x[:, :, i, :, :]
                    frame = func(self, frame, *args, **kwargs)
                    result.append(frame.unsqueeze(2))
                x = torch.concatenate(result, dim=2)
        return x

    return wrapper


class HaarWaveletTransform3D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        h = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]) * 0.3536
        g = torch.tensor([[[1, -1], [1, -1]], [[1, -1], [1, -1]]]) * 0.3536
        hh = torch.tensor([[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]]) * 0.3536
        gh = torch.tensor([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]]) * 0.3536
        h_v = torch.tensor([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]]) * 0.3536
        g_v = torch.tensor([[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]]) * 0.3536
        hh_v = torch.tensor([[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]]) * 0.3536
        gh_v = torch.tensor([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]) * 0.3536
        h = h.view(1, 1, 2, 2, 2)
        g = g.view(1, 1, 2, 2, 2)
        hh = hh.view(1, 1, 2, 2, 2)
        gh = gh.view(1, 1, 2, 2, 2)
        h_v = h_v.view(1, 1, 2, 2, 2)
        g_v = g_v.view(1, 1, 2, 2, 2)
        hh_v = hh_v.view(1, 1, 2, 2, 2)
        gh_v = gh_v.view(1, 1, 2, 2, 2)

        self.h_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.g_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.hh_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.gh_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.h_v_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.g_v_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.hh_v_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)
        self.gh_v_conv = CausalConv3d(1, 1, 2, padding=0, stride=2, bias=False)

        self.h_conv.conv.weight.data = h
        self.g_conv.conv.weight.data = g
        self.hh_conv.conv.weight.data = hh
        self.gh_conv.conv.weight.data = gh
        self.h_v_conv.conv.weight.data = h_v
        self.g_v_conv.conv.weight.data = g_v
        self.hh_v_conv.conv.weight.data = hh_v
        self.gh_v_conv.conv.weight.data = gh_v
        self.h_conv.requires_grad_(False)
        self.g_conv.requires_grad_(False)
        self.hh_conv.requires_grad_(False)
        self.gh_conv.requires_grad_(False)
        self.h_v_conv.requires_grad_(False)
        self.g_v_conv.requires_grad_(False)
        self.hh_v_conv.requires_grad_(False)
        self.gh_v_conv.requires_grad_(False)

    def forward(self, x):
        assert x.dim() == 5

        if torch_npu is not None:
            dtype = x.dtype
            x = x.to(npu_config.conv_dtype)
            self.to(npu_config.conv_dtype)

        b = x.shape[0]
        x = rearrange(x, "b c t h w -> (b c) 1 t h w")
        low_low_low = self.h_conv(x)
        low_low_low = rearrange(low_low_low, "(b c) 1 t h w -> b c t h w", b=b)
        low_low_high = self.g_conv(x)
        low_low_high = rearrange(low_low_high, "(b c) 1 t h w -> b c t h w", b=b)
        low_high_low = self.hh_conv(x)
        low_high_low = rearrange(low_high_low, "(b c) 1 t h w -> b c t h w", b=b)
        low_high_high = self.gh_conv(x)
        low_high_high = rearrange(low_high_high, "(b c) 1 t h w -> b c t h w", b=b)
        high_low_low = self.h_v_conv(x)
        high_low_low = rearrange(high_low_low, "(b c) 1 t h w -> b c t h w", b=b)
        high_low_high = self.g_v_conv(x)
        high_low_high = rearrange(high_low_high, "(b c) 1 t h w -> b c t h w", b=b)
        high_high_low = self.hh_v_conv(x)
        high_high_low = rearrange(high_high_low, "(b c) 1 t h w -> b c t h w", b=b)
        high_high_high = self.gh_v_conv(x)
        high_high_high = rearrange(high_high_high, "(b c) 1 t h w -> b c t h w", b=b)

        output = torch.cat(
            [
                low_low_low,
                low_low_high,
                low_high_low,
                low_high_high,
                high_low_low,
                high_low_high,
                high_high_low,
                high_high_high,
            ],
            dim=1,
        )

        if torch_npu is not None:
            x = x.to(dtype)
            output = output.to(dtype)
            self.to(dtype)

        return output


class InverseHaarWaveletTransform3D(nn.Module):
    def __init__(self, enable_cached=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.register_buffer("h", torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536)
        self.register_buffer("g", torch.tensor([[[1, -1], [1, -1]], [[1, -1], [1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536)
        self.register_buffer("hh", torch.tensor([[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536)
        self.register_buffer("gh", torch.tensor([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536)
        self.register_buffer("h_v", torch.tensor([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536)
        self.register_buffer("g_v", torch.tensor([[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536)
        self.register_buffer(
            "hh_v", torch.tensor([[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer(
            "gh_v", torch.tensor([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.enable_cached = enable_cached
        self.causal_cached = None

    def forward(self, coeffs):
        assert coeffs.dim() == 5

        if torch_npu is not None:
            dtype = coeffs.dtype
            coeffs = coeffs.to(npu_config.conv_dtype)
            self.h = self.h.to(npu_config.conv_dtype)
            self.g = self.g.to(npu_config.conv_dtype)
            self.hh = self.hh.to(npu_config.conv_dtype)
            self.gh = self.gh.to(npu_config.conv_dtype)
            self.h_v = self.h_v.to(npu_config.conv_dtype)
            self.g_v = self.g_v.to(npu_config.conv_dtype)
            self.hh_v = self.hh_v.to(npu_config.conv_dtype)
            self.gh_v = self.gh_v.to(npu_config.conv_dtype)

        b = coeffs.shape[0]

        (
            low_low_low,
            low_low_high,
            low_high_low,
            low_high_high,
            high_low_low,
            high_low_high,
            high_high_low,
            high_high_high,
        ) = coeffs.chunk(8, dim=1)

        low_low_low = rearrange(low_low_low, "b c t h w -> (b c) 1 t h w")
        low_low_high = rearrange(low_low_high, "b c t h w -> (b c) 1 t h w")
        low_high_low = rearrange(low_high_low, "b c t h w -> (b c) 1 t h w")
        low_high_high = rearrange(low_high_high, "b c t h w -> (b c) 1 t h w")
        high_low_low = rearrange(high_low_low, "b c t h w -> (b c) 1 t h w")
        high_low_high = rearrange(high_low_high, "b c t h w -> (b c) 1 t h w")
        high_high_low = rearrange(high_high_low, "b c t h w -> (b c) 1 t h w")
        high_high_high = rearrange(high_high_high, "b c t h w -> (b c) 1 t h w")

        low_low_low = F.conv_transpose3d(low_low_low, self.h, stride=2)
        low_low_high = F.conv_transpose3d(low_low_high, self.g, stride=2)
        low_high_low = F.conv_transpose3d(low_high_low, self.hh, stride=2)
        low_high_high = F.conv_transpose3d(low_high_high, self.gh, stride=2)
        high_low_low = F.conv_transpose3d(high_low_low, self.h_v, stride=2)
        high_low_high = F.conv_transpose3d(high_low_high, self.g_v, stride=2)
        high_high_low = F.conv_transpose3d(high_high_low, self.hh_v, stride=2)
        high_high_high = F.conv_transpose3d(high_high_high, self.gh_v, stride=2)
        if self.enable_cached and self.causal_cached:
            reconstructed = (
                low_low_low
                + low_low_high
                + low_high_low
                + low_high_high
                + high_low_low
                + high_low_high
                + high_high_low
                + high_high_high
            )
        else:
            reconstructed = (
                low_low_low[:, :, 1:]
                + low_low_high[:, :, 1:]
                + low_high_low[:, :, 1:]
                + low_high_high[:, :, 1:]
                + high_low_low[:, :, 1:]
                + high_low_high[:, :, 1:]
                + high_high_low[:, :, 1:]
                + high_high_high[:, :, 1:]
            )
            self.causal_cached = True
        reconstructed = rearrange(reconstructed, "(b c) 1 t h w -> b c t h w", b=b)

        if torch_npu is not None:
            coeffs = coeffs.to(dtype)
            reconstructed = reconstructed.to(dtype)
            self.h = self.h.to(dtype)
            self.g = self.g.to(dtype)
            self.hh = self.hh.to(dtype)
            self.gh = self.gh.to(dtype)
            self.h_v = self.h_v.to(dtype)
            self.g_v = self.g_v.to(dtype)
            self.hh_v = self.hh_v.to(dtype)
            self.gh_v = self.gh_v.to(dtype)

        return reconstructed


class HaarWaveletTransform2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("aa", torch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer("ad", torch.tensor([[1, 1], [-1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer("da", torch.tensor([[1, -1], [1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer("dd", torch.tensor([[1, -1], [-1, 1]]).view(1, 1, 2, 2) / 2)

    @video_to_image
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * c, 1, h, w)
        low_low = F.conv2d(x, self.aa, stride=2).reshape(b, c, h // 2, w // 2)
        low_high = F.conv2d(x, self.ad, stride=2).reshape(b, c, h // 2, w // 2)
        high_low = F.conv2d(x, self.da, stride=2).reshape(b, c, h // 2, w // 2)
        high_high = F.conv2d(x, self.dd, stride=2).reshape(b, c, h // 2, w // 2)
        coeffs = torch.cat([low_low, low_high, high_low, high_high], dim=1)
        return coeffs


class InverseHaarWaveletTransform2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("aa", torch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer("ad", torch.tensor([[1, 1], [-1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer("da", torch.tensor([[1, -1], [1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer("dd", torch.tensor([[1, -1], [-1, 1]]).view(1, 1, 2, 2) / 2)

    @video_to_image
    def forward(self, coeffs):
        low_low, low_high, high_low, high_high = coeffs.chunk(4, dim=1)
        b, c, height_half, width_half = low_low.shape
        height = height_half * 2
        width = width_half * 2

        low_low = F.conv_transpose2d(low_low.reshape(b * c, 1, height_half, width_half), self.aa, stride=2)
        low_high = F.conv_transpose2d(low_high.reshape(b * c, 1, height_half, width_half), self.ad, stride=2)
        high_low = F.conv_transpose2d(high_low.reshape(b * c, 1, height_half, width_half), self.da, stride=2)
        high_high = F.conv_transpose2d(high_high.reshape(b * c, 1, height_half, width_half), self.dd, stride=2)

        return (low_low + low_high + high_low + high_high).reshape(b, c, height, width)
