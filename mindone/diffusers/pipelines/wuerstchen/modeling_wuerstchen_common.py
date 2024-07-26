import mindspore as ms
from mindspore import nn, ops

from ...models.activations import SiLU
from ...models.attention_processor import Attention
from ...models.normalization import LayerNorm


class WuerstchenLayerNorm(LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def construct(self, x):
        x = x.permute((0, 2, 3, 1))
        x = super().construct(x)
        return x.permute((0, 3, 1, 2))


class TimestepBlock(nn.Cell):
    def __init__(self, c, c_timestep):
        super().__init__()
        linear_cls = nn.Dense
        self.mapper = linear_cls(c_timestep, c * 2)

    def construct(self, x, t):
        a, b = self.mapper(t)[:, :, None, None].chunk(2, axis=1)
        return x * (1 + a) + b


class ResBlock(nn.Cell):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):
        super().__init__()

        conv_cls = nn.Conv2d
        linear_cls = nn.Dense

        self.depthwise = conv_cls(
            c + c_skip, c, kernel_size=kernel_size, padding=kernel_size // 2, group=c, has_bias=True, pad_mode="pad"
        )
        self.norm = WuerstchenLayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.SequentialCell(
            linear_cls(c, c * 4), nn.GELU(), GlobalResponseNorm(c * 4), nn.Dropout(p=dropout), linear_cls(c * 4, c)
        )

    def construct(self, x, x_skip=None):
        x_res = x
        if x_skip is not None:
            x = ops.cat([x, x_skip], axis=1)
        x = self.norm(self.depthwise(x)).permute((0, 2, 3, 1))
        x = self.channelwise(x).permute((0, 3, 1, 2))
        return x + x_res


# from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
class GlobalResponseNorm(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.gamma = ms.Parameter(ops.zeros((1, 1, 1, dim)), name="gamma")
        self.beta = ms.Parameter(ops.zeros((1, 1, 1, dim)), name="beta")

    def construct(self, x):
        agg_norm = ops.norm(x, ord="fro", dim=(1, 2), keepdim=True)
        stand_div_norm = agg_norm / (agg_norm.mean(axis=-1, keep_dims=True) + 1e-6)
        return self.gamma * (x * stand_div_norm) + self.beta + x


class AttnBlock(nn.Cell):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()

        linear_cls = nn.Dense

        self.self_attn = self_attn
        self.norm = WuerstchenLayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(query_dim=c, heads=nhead, dim_head=c // nhead, dropout=dropout, bias=True)
        self.kv_mapper = nn.SequentialCell(SiLU(), linear_cls(c_cond, c))

    def construct(self, x, kv):
        kv = self.kv_mapper(kv)
        norm_x = self.norm(x)
        if self.self_attn:
            batch_size, channel, _, _ = x.shape
            kv = ops.cat([norm_x.view(batch_size, channel, -1).transpose(0, 2, 1), kv], axis=1)
        x = x + self.attention(norm_x, encoder_hidden_states=kv)
        return x
