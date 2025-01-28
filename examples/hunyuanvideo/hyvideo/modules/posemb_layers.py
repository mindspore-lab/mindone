from typing import List, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import ops

#################################################################################
#                   Rotary Positional Embedding Functions                       #
#################################################################################
# https://github.com/meta-llama/llama/blob/be327c427cc5e89cc1d3ab3d3fec4484df771245/llama/model.py#L80


def view_as_complex(input: ms.Tensor) -> ms.Tensor:
    # assert input.shape[-1] == 2, "Tensor must have a last dimension of size 2"
    real_part, imag_part = input.chunk(2, axis=-1)
    output = ops.Complex()(real_part, imag_part).squeeze(axis=-1)
    return output


def reshape_for_broadcast(
    freqs_cis: Union[ms.Tensor, Tuple[ms.Tensor]],
    x: ms.Tensor,
    head_first=False,
):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Notes:
        When using FlashMHAModified, head_first should be False.
        When using Attention, head_first should be True.

    Args:
        freqs_cis (Union[ms.Tensor, Tuple[ms.Tensor]]): Frequency tensor to be reshaped.
        x (ms.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        ms.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    # ndim = x.ndim
    # assert 0 <= 1 < ndim

    # x: (B S H D)
    # freqs_cis[0] or freqs_cis: (S D)
    # shape = (1 S 1 D)
    _, S, _, D = x.shape
    shape = (1, S, 1, D)  # [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        # assert freqs_cis[0].shape == (
        #    x.shape[1],
        #    x.shape[-1],
        # ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"

        return freqs_cis[0].reshape(shape), freqs_cis[1].reshape(shape)
    else:
        # assert freqs_cis.shape == (
        #    x.shape[1],
        #    x.shape[-1],
        # ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
        # shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.reshape(shape)


def rotate_half(x):
    # [B, S, H, D] -> [B, S, H, D//2, 2]
    x = x.reshape(x.shape[:-1] + (-1, 2))
    # real/image: [B, S, H, D//2, 1]
    x_real, x_imag = ops.chunk(x, 2, axis=-1)
    # [B, S, H, D//2, 2]
    x_out = ops.concat((-x_imag, x_real), axis=-1)

    # ret [B, S, H, D]
    return x_out.reshape(x_out.shape[:-2] + (-1,))


def apply_rotary_emb(
    xq: ms.Tensor,
    xk: ms.Tensor,
    freqs_cis: Union[ms.Tensor, Tuple[ms.Tensor, ms.Tensor]],
    head_first: bool = False,
) -> Tuple[ms.Tensor, ms.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (ms.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (ms.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (ms.Tensor or tuple): Precomputed frequency tensor for complex exponential.
            can be a complex tensor or a tuple of two tensors (cos, sin) representing a complex
        head_first (bool): head dimension first (except batch dim) or not. (true if FA use BNSD format for better speed). not supported.

    Returns:
        Tuple[ms.Tensor, ms.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    xq_dtype = xq.dtype
    xk_dtype = xk.dtype
    if isinstance(freqs_cis, tuple):
        # [S, D] -> [1, S, 1, D]
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_out = (xq.to(ms.float32) * cos + rotate_half(xq.to(ms.float32)) * sin).to(xq_dtype)
        xk_out = (xk.to(ms.float32) * cos + rotate_half(xk.to(ms.float32)) * sin).to(xk_dtype)
    else:
        # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
        xq_ = view_as_complex(xq.float().reshape(xq.shape[:-1] + (-1, 2)))  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first)  # [S, D//2] --> [1, S, 1, D//2]
        # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
        # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
        xq_out = ops.view_as_real(xq_ * freqs_cis).flatten(start_dim=3).to(xq_dtype)

        xk_ = view_as_complex(xk.float().reshape(xk.shape[:-1] + (-1, 2)))  # [B, S, H, D//2]
        xk_out = ops.view_as_real(xk_ * freqs_cis).flatten(start_dim=3).to(xk_dtype)

    return xq_out, xk_out


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (np.ndarray): [dim, ...]
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        # a, b, n = start[i], stop[i], num[i]
        # g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        g = np.linspace(start[i], stop[i], num[i] + 1, dtype=np.float32)[: num[i]]
        axis_grid.append(g)

    grid = np.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = np.stack(grid, axis=0)  # [dim, W, H, D]
    grid = ms.Tensor(grid)

    return grid


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
):
    """
    This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
            part and an imaginary part separately.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.

    Returns:
        pos_embed (torch.Tensor): [HW, D/2]
    """
    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))  # [3, W, H, D] / [2, W, H]

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(
        rope_dim_list
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(
        rope_dim_list
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )  # 2 x [WHD, rope_dim_list[i]]
        embs.append(emb)

    if use_real:
        cos = ops.cat([emb[0] for emb in embs], axis=1)  # (WHD, D/2)
        sin = ops.cat([emb[1] for emb in embs], axis=1)  # (WHD, D/2)
        return cos, sin
    else:
        emb = ops.cat(embs, axis=1)  # (WHD, D/2)
        return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[ms.Tensor, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Union[ms.Tensor, Tuple[ms.Tensor, ms.Tensor]]:
    """
    Precompute the frequency tensor for complex exponential (cis) with given dimensions.
    (Note: `cis` means `cos + i * sin`, where i is the imaginary unit.)

    This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (int or torch.FloatTensor): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.
        theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.

    Returns:
        freqs_cis: Precomputed frequency tensor with complex exponential. [S, D/2]
        freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
    """
    if isinstance(pos, int):
        pos = ops.arange(pos).float()

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (theta ** (ops.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [D/2]
    # assert interpolation_factor == 1.0, f"interpolation_factor: {interpolation_factor}"
    # TODO: outer doesn't support broadcasting
    freqs = ops.outer(pos * interpolation_factor, freqs)  # [S, D/2]
    if use_real:
        # TODO: check ms2.5 API for repeat_interleave
        freqs_cos = ops.repeat_interleave(freqs.cos(), repeats=2, axis=1)  # [S, D]
        freqs_sin = ops.repeat_interleave(freqs.sin(), repeats=2, axis=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = ops.polar(ops.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis
