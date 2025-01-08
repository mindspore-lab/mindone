from functools import partial
from mindspore import nn

from ..utils.helpers import to_2tuple


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
        # TODO: doing here
        if use_conv:
            linear_layer = partial(nn.Conv2d, kernel_size=1, pad_mode='valid', has_bias=True)
        else:
            linear_layer = nn.Dense

        self.fc1 = linear_layer(
            in_channels, hidden_channels, has_bias=bias[0],
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop_probs[0])
        self.norm = (
            norm_layer(hidden_channels, **factory_kwargs)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = linear_layer(
            hidden_channels, out_features, has_bias=bias[1],
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

