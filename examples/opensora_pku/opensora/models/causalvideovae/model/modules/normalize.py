from mindspore import nn

from mindone.diffusers.models.normalization import LayerNorm as LayerNorm_diffusers


# TODO: put them to modules/normalize.py
class GroupNormExtend(nn.GroupNorm):
    # GroupNorm supporting tensors with more than 4 dim
    def construct(self, x):
        x_shape = x.shape
        if x.ndim >= 5:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
        y = super().construct(x)
        return y.view(x_shape)


class LayerNorm(nn.Cell):
    def __init__(self, num_channels, eps=1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = LayerNorm_diffusers(num_channels, eps=eps, elementwise_affine=True)

    def construct(self, x):
        if x.ndim == 5:
            # b c t h w -> b t h w c
            x = x.transpose(0, 2, 3, 4, 1)
            x = self.norm(x)
            # b t h w c -> b c t h w
            x = x.transpose(0, 4, 1, 2, 3)
        else:
            # b c h w -> b h w c
            x = x.transpose(0, 2, 3, 1)
            x = self.norm(x)
            # b h w c -> b c h w
            x = x.transpose(0, 3, 1, 2)
        return x


def Normalize(in_channels, num_groups=32, norm_type="groupnorm"):
    if norm_type == "groupnorm":
        return GroupNormExtend(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == "layernorm":
        return LayerNorm(num_channels=in_channels, eps=1e-6)
