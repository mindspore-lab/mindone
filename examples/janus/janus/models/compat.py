from packaging.version import parse

import mindspore as ms
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.common.initializer import initializer


def normalize_l2(
    input: Tensor,
    dim: int = 1,
    eps: float = 1e-12,
) -> Tensor:
    r"""Perform :math:`L_p` normalization of inputs over specified dimension.

    Equivalent to torch.nn.functional.normalize with fixed p=2

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.

    Args:
        input: input tensor of any shape
        dim (int or tuple of ints): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
    """
    l2norm = ops.norm(input, ord=None, dim=dim, keepdim=True)
    # l2norm = mint.sqrt(mint.sum(mint.square(input), dim=dim, keepdim=True))
    denom = ops.clamp(l2norm, min=eps).expand_as(input)
    return ops.div(input, denom)


# TODO: same accuracy as mint.nn.GroupNorm
group_norm = ms.mint.nn.functional.group_norm


class GroupNorm(nn.Cell):
    # gamma -> weight, beta -> bias
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        dtype=ms.float32,
    ):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        weight = initializer("ones", num_channels, dtype=dtype)
        bias = initializer("zeros", num_channels, dtype=dtype)
        if self.affine:
            self.weight = Parameter(weight, name="weight")
            self.bias = Parameter(bias, name="bias")
        else:
            self.weight = None
            self.bias = None

    def construct(self, x: Tensor):
        if self.affine:
            x = group_norm(
                x,
                self.num_groups,
                self.weight.to(x.dtype),
                self.bias.to(x.dtype),
                self.eps,
            )
        else:
            x = group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        return x


def get_multinomial_op():
    if parse(ms.__version__) >= parse("2.5"):
        return mint.multinomial
    else:
        if ms.get_context("mode") == 0:
            return ops.multinomial
        else:
            # before ms2.5, mint multinomial doesn't support graph mode
            return mint.multinomial
