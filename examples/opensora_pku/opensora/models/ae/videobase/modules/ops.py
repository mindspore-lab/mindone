import mindspore as ms
from mindspore import ops


def nonlinearity(x, upcast=False):
    # swish
    ori_dtype = x.dtype
    if upcast:
        return x * (ops.sigmoid(x.astype(ms.float32))).astype(ori_dtype)
    else:
        return x * (ops.sigmoid(x))


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def shift_dim(x, src_dim=-1, dest_dim=-1):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim
    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims
    dims = list(range(n_dims))
    del dims[src_dim]
    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)

    return x
