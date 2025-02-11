import mindspore as ms
from mindspore import mint


def video_to_image(func):
    def wrapper(self, x, *args, **kwargs):
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            if True:
                # b c t h w -> (b t) c h w
                x = x.swapaxes(1, 2).reshape(-1, c, h, w)  # (b*t, c, h, w)
                x = func(self, x, *args, **kwargs)
                x = x.reshape(x.shape[0] // t, t, x.shape[1], x.shape[2], x.shape[3])  # (b, t, c, h, w)
                x = x.transpose(0, 2, 1, 3, 4)  # (b, c, t, h, w)
            else:
                # Conv 2d slice infer
                result = []
                for i in range(t):
                    frame = x[:, :, i, :, :]
                    frame = func(self, frame, *args, **kwargs)
                    result.append(frame.unsqueeze(2))
                x = mint.cat(result, dim=2)
        return x

    return wrapper


def nonlinearity(x, upcast=False):
    # swish
    ori_dtype = x.dtype
    if upcast:
        return x * (mint.sigmoid(x.astype(ms.float32))).astype(ori_dtype)
    else:
        return x * (mint.sigmoid(x))
    # return nn.SiLU()(x.astype(ms.float32) if upcast else x).to(ori_dtype)


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
