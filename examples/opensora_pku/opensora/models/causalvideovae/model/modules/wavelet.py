import logging

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

from ..modules import CausalConv3d
from ..modules.ops import video_to_image

try:
    from opensora.npu_config import npu_config
except ImportError:
    npu_config = None

logger = logging.getLogger(__name__)


class HaarWaveletTransform3D(nn.Cell):
    def __init__(self, dtype=ms.float32) -> None:
        super().__init__()
        self.dtype = dtype
        h = Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]) * 0.3536
        g = Tensor([[[1, -1], [1, -1]], [[1, -1], [1, -1]]]) * 0.3536
        hh = Tensor([[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]]) * 0.3536
        gh = Tensor([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]]) * 0.3536
        h_v = Tensor([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]]) * 0.3536
        g_v = Tensor([[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]]) * 0.3536
        hh_v = Tensor([[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]]) * 0.3536
        gh_v = Tensor([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]) * 0.3536
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

        self.h_conv.conv.weight.set_data(h)
        self.g_conv.conv.weight.set_data(g)
        self.hh_conv.conv.weight.set_data(hh)
        self.gh_conv.conv.weight.set_data(gh)
        self.h_v_conv.conv.weight.set_data(h_v)
        self.g_v_conv.conv.weight.set_data(g_v)
        self.hh_v_conv.conv.weight.set_data(hh_v)
        self.gh_v_conv.conv.weight.set_data(gh_v)
        self.h_conv.requires_grad = False
        self.g_conv.requires_grad = False
        self.hh_conv.requires_grad = False
        self.gh_conv.requires_grad = False
        self.h_v_conv.requires_grad = False
        self.g_v_conv.requires_grad = False
        self.hh_v_conv.requires_grad = False
        self.gh_v_conv.requires_grad = False

    def construct(self, x):
        assert x.ndim == 5

        b = x.shape[0]
        # b c t h w -> (b c) 1 t h w
        x = x.reshape(-1, 1, *x.shape[-3:])
        low_low_low = self.h_conv(x)
        low_low_low = low_low_low.reshape(
            b, low_low_low.shape[0] // b, low_low_low.shape[-3], low_low_low.shape[-2], low_low_low.shape[-1]
        )  # (b c) 1 t h w -> b c t h w

        low_low_high = self.g_conv(x)
        low_low_high = low_low_high.reshape(
            b, low_low_high.shape[0] // b, low_low_high.shape[-3], low_low_high.shape[-2], low_low_high.shape[-1]
        )  # (b c) 1 t h w -> b c t h w

        low_high_low = self.hh_conv(x)
        low_high_low = low_high_low.reshape(
            b, low_high_low.shape[0] // b, low_high_low.shape[-3], low_high_low.shape[-2], low_high_low.shape[-1]
        )  # (b c) 1 t h w -> b c t h w

        low_high_high = self.gh_conv(x)
        low_high_high = low_high_high.reshape(
            b, low_high_high.shape[0] // b, low_high_high.shape[-3], low_high_high.shape[-2], low_high_high.shape[-1]
        )  # (b c) 1 t h w -> b c t h w

        high_low_low = self.h_v_conv(x)
        high_low_low = high_low_low.reshape(
            b, high_low_low.shape[0] // b, high_low_low.shape[-3], high_low_low.shape[-2], high_low_low.shape[-1]
        )  # (b c) 1 t h w -> b c t h w

        high_low_high = self.g_v_conv(x)
        high_low_high = high_low_high.reshape(
            b, high_low_high.shape[0] // b, high_low_high.shape[-3], high_low_high.shape[-2], high_low_high.shape[-1]
        )  # (b c) 1 t h w -> b c t h w

        high_high_low = self.hh_v_conv(x)
        high_high_low = high_high_low.reshape(
            b, high_high_low.shape[0] // b, high_high_low.shape[-3], high_high_low.shape[-2], high_high_low.shape[-1]
        )  # (b c) 1 t h w -> b c t h w

        high_high_high = self.gh_v_conv(x)
        high_high_high = high_high_high.reshape(
            b,
            high_high_high.shape[0] // b,
            high_high_high.shape[-3],
            high_high_high.shape[-2],
            high_high_high.shape[-1],
        )  # (b c) 1 t h w -> b c t h w

        output = mint.cat(
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

        return output


class InverseHaarWaveletTransform3D(nn.Cell):
    def __init__(self, enable_cached=False, dtype=ms.float32, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dtype = dtype

        if self.dtype != ms.float16:
            # Conv3dTranspose is forced to fp16
            self.dtype = ms.float16
            dtype = ms.float16

        self.h = Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=dtype).view(1, 1, 2, 2, 2) * 0.3536
        self.g = Tensor([[[1, -1], [1, -1]], [[1, -1], [1, -1]]], dtype=dtype).view(1, 1, 2, 2, 2) * 0.3536
        self.hh = Tensor([[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]], dtype=dtype).view(1, 1, 2, 2, 2) * 0.3536
        self.gh = Tensor([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]], dtype=dtype).view(1, 1, 2, 2, 2) * 0.3536
        self.h_v = Tensor([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]], dtype=dtype).view(1, 1, 2, 2, 2) * 0.3536
        self.g_v = Tensor([[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]], dtype=dtype).view(1, 1, 2, 2, 2) * 0.3536
        self.hh_v = Tensor([[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]], dtype=dtype).view(1, 1, 2, 2, 2) * 0.3536
        self.gh_v = Tensor([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]], dtype=dtype).view(1, 1, 2, 2, 2) * 0.3536
        self.enable_cached = enable_cached
        self.is_first_chunk = True
        self.conv_transpose3d = ops.Conv3DTranspose(1, 1, kernel_size=2, stride=2)

    def construct(self, coeffs):
        assert coeffs.ndim == 5
        input_dtype = coeffs.dtype
        coeffs = coeffs.to(self.dtype)
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
        ) = mint.chunk(coeffs, 8, dim=1)

        low_low_low = low_low_low.reshape(-1, 1, *low_low_low.shape[-3:])
        low_low_high = low_low_high.reshape(-1, 1, *low_low_high.shape[-3:])
        low_high_low = low_high_low.reshape(-1, 1, *low_high_low.shape[-3:])
        low_high_high = low_high_high.reshape(-1, 1, *low_high_high.shape[-3:])
        high_low_low = high_low_low.reshape(-1, 1, *high_low_low.shape[-3:])
        high_low_high = high_low_high.reshape(-1, 1, *high_low_high.shape[-3:])
        high_high_low = high_high_low.reshape(-1, 1, *high_high_low.shape[-3:])
        high_high_high = high_high_high.reshape(-1, 1, *high_high_high.shape[-3:])

        low_low_low = self.conv_transpose3d(low_low_low, self.h)
        low_low_high = self.conv_transpose3d(low_low_high, self.g)
        low_high_low = self.conv_transpose3d(low_high_low, self.hh)
        low_high_high = self.conv_transpose3d(low_high_high, self.gh)
        high_low_low = self.conv_transpose3d(high_low_low, self.h_v)
        high_low_high = self.conv_transpose3d(high_low_high, self.g_v)
        high_high_low = self.conv_transpose3d(high_high_low, self.hh_v)
        high_high_high = self.conv_transpose3d(high_high_high, self.gh_v)
        if self.enable_cached and not self.is_first_chunk:
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

        reconstructed = reconstructed.reshape(b, -1, *reconstructed.shape[-3:])

        return reconstructed.to(input_dtype)


class HaarWaveletTransform2D(nn.Cell):
    def __init__(self, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.aa = Tensor([[1, 1], [1, 1]], dtype=dtype).view(1, 1, 2, 2) / 2
        self.ad = Tensor([[1, 1], [-1, -1]], dtype=dtype).view(1, 1, 2, 2) / 2
        self.da = Tensor([[1, -1], [1, -1]], dtype=dtype).view(1, 1, 2, 2) / 2
        self.dd = Tensor([[1, -1], [-1, 1]], dtype=dtype).view(1, 1, 2, 2) / 2

    @video_to_image
    def construct(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * c, 1, h, w)
        x_dtype = x.dtype
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        low_low = ops.conv2d(x, self.aa, stride=2).reshape(b, c, h // 2, w // 2)
        low_high = ops.conv2d(x, self.ad, stride=2).reshape(b, c, h // 2, w // 2)
        high_low = ops.conv2d(x, self.da, stride=2).reshape(b, c, h // 2, w // 2)
        high_high = ops.conv2d(x, self.dd, stride=2).reshape(b, c, h // 2, w // 2)
        coeffs = mint.cat([low_low, low_high, high_low, high_high], dim=1)
        return coeffs.to(x_dtype)


class InverseHaarWaveletTransform2D(nn.Cell):
    def __init__(self, dtype=ms.float32):
        super().__init__()
        aa = Tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2) / 2
        ad = Tensor([[1, 1], [-1, -1]]).view(1, 1, 2, 2) / 2
        da = Tensor([[1, -1], [1, -1]]).view(1, 1, 2, 2) / 2
        dd = Tensor([[1, -1], [-1, 1]]).view(1, 1, 2, 2) / 2
        self.dtype = dtype
        self.aa = nn.Conv2dTranspose(1, 1, kernel_size=2, stride=2, has_bias=False).to_float(dtype)
        self.ad = nn.Conv2dTranspose(1, 1, kernel_size=2, stride=2, has_bias=False).to_float(dtype)
        self.da = nn.Conv2dTranspose(1, 1, kernel_size=2, stride=2, has_bias=False).to_float(dtype)
        self.dd = nn.Conv2dTranspose(1, 1, kernel_size=2, stride=2, has_bias=False).to_float(dtype)
        self.aa.weight.set_data(aa)
        self.aa.requires_grad = False
        self.ad.weight.set_data(ad)
        self.ad.requires_grad = False
        self.da.weight.set_data(da)
        self.da.requires_grad = False
        self.dd.weight.set_data(dd)
        self.dd.requires_grad = False

    @video_to_image
    def construct(self, coeffs):
        low_low, low_high, high_low, high_high = mint.chunk(coeffs, 4, dim=1)
        b, c, height_half, width_half = low_low.shape
        height = height_half * 2
        width = width_half * 2

        low_low = self.aa(low_low.reshape(b * c, 1, height_half, width_half))
        low_high = self.ad(low_high.reshape(b * c, 1, height_half, width_half))
        high_low = self.da(high_low.reshape(b * c, 1, height_half, width_half))
        high_high = self.dd(high_high.reshape(b * c, 1, height_half, width_half))

        return (low_low + low_high + high_low + high_high).reshape(b, c, height, width)
