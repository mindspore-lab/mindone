import mindspore as ms
from mindspore import Parameter, nn, ops


def norm_except_dim(v, pow, dim):
    if dim == -1:
        return ops.norm(v, pow)
    elif dim == 0:
        output_size = (v.shape[0],) + (1,) * (v.ndim - 1)
        return ops.norm(v.view((v.shape[0], -1)), pow, 1).view(output_size)
    elif dim == (v.ndim - 1):
        output_size = (1,) * (v.ndim - 1) + (v.shape[v.ndim - 1])
        return ops.norm(v.view((-1, v.shape[v.ndim - 1])), pow, 0).view(output_size)
    else:
        return norm_except_dim(v.swapaxes(0, dim), pow, dim).swapaxes(0, dim)


def _weight_norm(v, g, dim):
    return v * (g / norm_except_dim(v, 2, dim))


class WeightNorm(nn.Cell):
    def __init__(self, weight_norm_cell, dim: int = 0):
        super().__init__()

        if dim is None:
            dim = -1

        self.dim = dim
        self.weight_norm_cell = weight_norm_cell

        # add g and v as new parameters and express w as g/||v|| * v
        self.weight_g = Parameter(ms.Tensor(norm_except_dim(self.weight_norm_cell.weight, 2, dim)))
        self.weight_v = Parameter(ms.Tensor(self.weight_norm_cell.weight.data))
        self.weight_norm_cell.weight.set_data(_weight_norm(self.weight_v, self.weight_g, self.dim))

        self.use_weight_norm = True
        self.kernel_size = weight_norm_cell.kernel_size
        self.stride = weight_norm_cell.stride
        self.dilation = weight_norm_cell.dilation

    def construct(self, *inputs, **kwargs):
        if not self.use_weight_norm:
            return self.weight_norm_cell(*inputs, **kwargs)

        ops.assign(self.weight_norm_cell.weight, _weight_norm(self.weight_v, self.weight_g, self.dim))
        return self.weight_norm_cell(*inputs, **kwargs)

    def remove_weight_norm(self):
        self.assign(self.weight_norm_cell.weight, _weight_norm(self.weight_v, self.weight_g, self.dim))
        self.use_weight_norm = False
