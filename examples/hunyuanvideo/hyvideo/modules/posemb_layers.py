import mindspore as ms
from mindspore import ops
from typing import Union, Tuple, List

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
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

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
    shape = (1, S, 1, D) # [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        # assert freqs_cis[0].shape == (
        #    x.shape[1],
        #    x.shape[-1],
        # ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"

        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # assert freqs_cis.shape == (
        #    x.shape[1],
        #    x.shape[-1],
        # ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
        # shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    # x_real, x_imag = (
    #    x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    #)  # [B, S, H, D//2]
    # return torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    # [B, S, H, D] -> [B, S, H, D//2, 2]
    x = x.reshape(x.shape[:-1] + (-1, 2))
    # real/image: [B, S, H, D//2, 1]
    x_real, x_imag = ops.chunk(x, 2, axis=-1)
    # [B, S, H, D//2, 2]
    x_out = ops.concat((-x_imag, x_real), axis=-1)

    # ret [B, S, H, D]
    return x_out.reshape(x_out.shape[:-2] + (-1,))


# TODO: for graph mode, may need to convert freqs_cis to one tensor
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
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (torch.Tensor or tuple): Precomputed frequency tensor for complex exponential. can be a complex tensor or a tuple of two tensors (cos, sin) representing a complex
        head_first (bool): head dimension first (except batch dim) or not. (true if FA use BNSD format for better speed). not supported.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

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
        xq_ = view_as_complex(
            xq.float().reshape(xq.shape[:-1] + (-1, 2))
        )  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first) # [S, D//2] --> [1, S, 1, D//2]
        # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
        # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
        xq_out = ops.view_as_real(xq_ * freqs_cis).flatten(start_dim=3).to(xq_dtype)

        xk_ = view_as_complex(
            xk.float().reshape(xk.shape[:-1] + (-1, 2))
        )  # [B, S, H, D//2]
        xk_out = ops.view_as_real(xk_ * freqs_cis).flatten(start_dim=3).to(xk_dtype)

    return xq_out, xk_out
