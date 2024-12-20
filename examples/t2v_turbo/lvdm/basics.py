# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!

from mindspore import nn, ops


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.get_parameters():
        p = ops.zeros_like(p)
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.get_parameters():
        p = p.mul(scale)
    return module


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Dense(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return ops.AvgPool3D(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def nonlinearity(type="silu"):
    if type == "silu":
        return nn.SiLU()
    elif type == "leaky_relu":
        return nn.LeakyReLU()


class GroupNormSpecific(nn.GroupNorm):
    def construct(self, x):
        return super().construct(x.float()).to(x.dtype)


def normalization(channels, num_groups=32):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Cell for normalization.
    """
    return GroupNormSpecific(num_groups, channels)


def rearrange_in_gn5d_bs(x, b):
    # (b*f c h w) -> (b f c h w) -> (b c f h w)
    bf, c, h, w = x.shape
    x = ops.reshape(x, (b, bf // b, c, h, w))
    x = ops.transpose(x, (0, 2, 1, 3, 4))

    return x


def rearrange_in_gn5d(x, video_length):
    # (b*f c h w) -> (b f c h w) -> (b c f h w) for GN5D
    bf, c, h, w = x.shape
    x = ops.reshape(x, (bf // video_length, video_length, c, h, w))
    x = ops.transpose(x, (0, 2, 1, 3, 4))

    return x


def rearrange_out_gn5d(x):
    # (b c f h w) -> (b f c h w) -> (b*f c h w)
    b, c, f, h, w = x.shape
    x = ops.transpose(x, (0, 2, 1, 3, 4))
    x = ops.reshape(x, (-1, c, h, w))

    return x
