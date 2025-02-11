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
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
    """

    denom = input.norm(p, dim, keepdim=True).clip(min=eps).broadcast_to(input.shape)

    return input / denom
