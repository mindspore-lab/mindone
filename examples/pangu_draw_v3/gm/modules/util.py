from mindspore import Tensor, ops


def linear(x, weight, bias):
    x_shape = x.shape
    if len(x_shape) != 2:
        x = x.reshape(-1, x_shape[-1])
    x = ops.MatMul(transpose_b=True)(x, weight)
    x = ops.bias_add(x, bias)
    if len(x_shape) != 2:
        out_shape = x_shape[:-1] + (x.shape[-1],)
        x = x.reshape(*out_shape)
    return x


def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
    r"""
    Equivalence of 'torch.nn.functional.normalize(input, p=2.0, dim=1, eps=eps)'
    """

    denom = input.norm(p, dim, keepdim=True).clip(min=eps).broadcast_to(input.shape)

    return input / denom
