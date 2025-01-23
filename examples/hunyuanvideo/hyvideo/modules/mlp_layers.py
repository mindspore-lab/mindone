from functools import partial

import mindspore as ms
from mindspore import mint, nn

from ..utils.helpers import to_2tuple
from .modulate_layers import modulate
from .norm_layers import LayerNorm


class MLP(nn.Cell):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        if use_conv:
            linear_layer = partial(nn.Conv2d, kernel_size=1, pad_mode="valid", has_bias=True)
        else:
            linear_layer = mint.nn.Linear

        self.fc1 = linear_layer(
            in_channels,
            hidden_channels,
            bias=bias[0],
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop_probs[0])
        self.norm = norm_layer(hidden_channels, **factory_kwargs) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(
            hidden_channels,
            out_features,
            bias=bias[1],
        )
        self.drop2 = nn.Dropout(p=drop_probs[1])

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MLPEmbedder(nn.Cell):
    def __init__(self, in_dim: int, hidden_dim: int, dtype=None):
        # factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.in_layer = mint.nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = mint.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class FinalLayer(nn.Cell):
    """The final layer of DiT."""

    def __init__(self, hidden_size, patch_size, out_channels, act_layer, dtype=None):
        factory_kwargs = {"dtype": dtype}
        super().__init__()

        # Just use LayerNorm for the final layer
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        if isinstance(patch_size, int):
            self.linear = mint.nn.Linear(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
                weight_init="zeros",
                bias_init="zeros",
            )
        else:
            self.linear = mint.nn.Linear(
                hidden_size,
                patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
                bias=True,
                weight_init="zeros",
                bias_init="zeros",
            )

        # Here we don't distinguish between the modulate types. Just use the simple one.
        self.adaLN_modulation = nn.SequentialCell(
            act_layer(),
            mint.nn.Linear(hidden_size, 2 * hidden_size, bias=True, weight_init="zeros", bias_init="zeros"),
        )

    def construct(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, axis=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x
