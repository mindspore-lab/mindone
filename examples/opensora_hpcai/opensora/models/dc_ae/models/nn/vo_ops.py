import math
from inspect import signature
from typing import Any, Callable, Optional, Union

import mindspore as ms
from mindspore import mint

VERBOSE = False


def pixel_shuffle_3d(x, upscale_factor):
    """
    3D pixelshuffle
    """
    B, C, T, H, W = x.shape
    r = upscale_factor
    assert C % (r * r * r) == 0, f"C must be a cubic multiple of the upscale_factor, but got C: {C}, upscale_factor: {r}"

    C_new = C // (r * r * r)
    x = x.view(B, C_new, r, r, r, T, H, W)
    if VERBOSE:
        print("x.view:")
        print(x)
        print("x.view.shape:")
        print(x.shape)

    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    if VERBOSE:
        print("x.permute:")
        print(x)
        print("x.permute.shape:")
        print(x.shape)

    y = x.reshape(B, C_new, T * r, H * r, W * r)
    return y


def pixel_unshuffle_3d(x, downsample_factor):
    """
    3D pixel unshuffle
    """
    B, C, T, H, W = x.shape

    r = downsample_factor
    assert T % r == 0, f"T must be a multiple of the downsample_factor, but got T: {T}, downsample_factor: {r}"
    assert H % r == 0, f"H must be a multiple of the downsample_factor, but got H: {H}, downsample_factor: {r}"
    assert W % r == 0, f"W must be a multiple of the downsample_factor, but got W: {W}, downsample_factor: {r}"
    T_new = T // r
    H_new = H // r
    W_new = W // r
    C_new = C * (r * r * r)

    x = x.view(B, C, T_new, r, H_new, r, W_new, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    y = x.reshape(B, C_new, T_new, H_new, W_new)
    return y


def test_pixel_shuffle_3d():
    # input tensor (B, C, T, H, W) = (1, 16, 2, 4, 4)
    x = mint.arange(1, 1 + 1 * 16 * 2 * 4 * 4).view(1, 16, 2, 4, 4).float()
    print("x:")
    print(x)
    print("x.shape:")
    print(x.shape)

    upscale_factor = 2

    y = pixel_shuffle_3d(x, upscale_factor)
    print("pixelshuffle_3d result:")
    print(y)
    print("output shape:", y.shape)
    # Expected output shape: (1, 1, 4, 8, 8)
    # - C: 8->1 (8 /(2*2*2))
    # - T: 2->4 (2*2)
    # - H: 4->8 (4*2)
    # - W: 4->8 (4*2)

    print(mint.allclose(x, pixel_unshuffle_3d(y, upscale_factor)))


def chunked_interpolate(x, scale_factor, mode="nearest"):
    """
    Interpolate large tensors by chunking along the channel dimension. https://discuss.pytorch.org/t/error-using-f-interpolate-for-large-3d-input/207859
    Only supports 'nearest' interpolation mode.

    Args:
        x (ms.tensor): Input tensor (B, C, D, H, W)
        scale_factor: Tuple of scaling factors (d, h, w)

    Returns:
        ms.tensor: Interpolated tensor
    """
    assert (
        mode == "nearest"
    ), "Only the nearest mode is supported"  # actually other modes are theoretically supported but not tested
    if len(x.shape) != 5:
        raise ValueError("Expected 5D input tensor (B, C, D, H, W)")

    # Calculate max chunk size to avoid int32 overflow. num_elements < max_int32
    # Max int32 is 2^31 - 1
    max_elements_per_chunk = 2**31 - 1

    # Calculate output spatial dimensions
    out_d = math.ceil(x.shape[2] * scale_factor[0])
    out_h = math.ceil(x.shape[3] * scale_factor[1])
    out_w = math.ceil(x.shape[4] * scale_factor[2])

    # Calculate max channels per chunk to stay under limit
    elements_per_channel = out_d * out_h * out_w
    max_channels = max_elements_per_chunk // (x.shape[0] * elements_per_channel)

    # Use smaller of max channels or input channels
    chunk_size = min(max_channels, x.shape[1])

    # Ensure at least 1 channel per chunk
    chunk_size = max(1, chunk_size)
    if VERBOSE:
        print(f"Input channels: {x.shape[1]}")
        print(f"Chunk size: {chunk_size}")
        print(f"max_channels: {max_channels}")
        print(f"num_chunks: {math.ceil(x.shape[1] / chunk_size)}")

    chunks = []
    for i in range(0, x.shape[1], chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, x.shape[1])

        chunk = x[:, start_idx:end_idx, :, :, :]

        interpolated_chunk = mint.nn.functional.interpolate(chunk, scale_factor=scale_factor, mode="nearest")

        chunks.append(interpolated_chunk)

    if not chunks:
        raise ValueError(f"No chunks were generated. Input shape: {x.shape}")

    # Concatenate chunks along channel dimension
    return mint.cat(chunks, dim=1)


def test_chunked_interpolate():
    # Test case 1: Basic upscaling with scale_factor
    x1 = mint.randn(2, 16, 16, 32, 32)
    scale_factor = (2.0, 2.0, 2.0)
    assert mint.allclose(
        chunked_interpolate(x1, scale_factor=scale_factor), mint.nn.functional.interpolate(x1, scale_factor=scale_factor, mode="nearest")
    )

    # Test case 3: Downscaling with scale_factor
    x3 = mint.randn(2, 16, 32, 64, 64)
    scale_factor = (0.5, 0.5, 0.5)
    assert mint.allclose(
        chunked_interpolate(x3, scale_factor=scale_factor), mint.nn.functional.interpolate(x3, scale_factor=scale_factor, mode="nearest")
    )

    # Test case 4: Different scales per dimension
    x4 = mint.randn(2, 16, 16, 32, 32)
    scale_factor = (2.0, 1.5, 1.5)
    assert mint.allclose(
        chunked_interpolate(x4, scale_factor=scale_factor), mint.nn.functional.interpolate(x4, scale_factor=scale_factor, mode="nearest")
    )

    # Test case 5: Large input tensor
    x5 = mint.randn(2, 16, 64, 128, 128)
    scale_factor = (2.0, 2.0, 2.0)
    assert mint.allclose(
        chunked_interpolate(x5, scale_factor=scale_factor), mint.nn.functional.interpolate(x5, scale_factor=scale_factor, mode="nearest")
    )

    # Test case 7: Chunk size equal to input depth
    x7 = mint.randn(2, 16, 8, 32, 32)
    scale_factor = (2.0, 2.0, 2.0)
    assert mint.allclose(
        chunked_interpolate(x7, scale_factor=scale_factor), mint.nn.functional.interpolate(x7, scale_factor=scale_factor, mode="nearest")
    )

    # Test case 8: Single channel input
    x8 = mint.randn(2, 1, 16, 32, 32)
    scale_factor = (2.0, 2.0, 2.0)
    assert mint.allclose(
        chunked_interpolate(x8, scale_factor=scale_factor), mint.nn.functional.interpolate(x8, scale_factor=scale_factor, mode="nearest")
    )

    # Test case 9: Minimal batch size
    x9 = mint.randn(1, 16, 32, 64, 64)
    scale_factor = (0.5, 0.5, 0.5)
    assert mint.allclose(
        chunked_interpolate(x9, scale_factor=scale_factor), mint.nn.functional.interpolate(x9, scale_factor=scale_factor, mode="nearest")
    )

    # Test case 10: Non-power-of-2 dimensions
    x10 = mint.randn(2, 16, 15, 31, 31)
    scale_factor = (2.0, 2.0, 2.0)
    assert mint.allclose(
        chunked_interpolate(x10, scale_factor=scale_factor),
        mint.nn.functional.interpolate(x10, scale_factor=scale_factor, mode="nearest"),
    )

    # Test case 11: large output tensor


def get_same_padding(kernel_size: Union[int, tuple[int, ...]]) -> Union[int, tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def resize(
    x: ms.tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[list[float]] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
) -> ms.tensor:
    if mode in {"bilinear", "bicubic"}:
        return mint.nn.functional.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return mint.nn.functional.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


def build_kwargs_from_config(config: dict, target_func: Callable) -> dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


if __name__ == "__main__":
    test_chunked_interpolate()
