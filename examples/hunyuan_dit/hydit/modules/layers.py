from functools import partial

from mindspore import mint, nn


class Mlp(nn.Cell):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=mint.nn.functional.gelu,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Dense

        self.fc1 = linear_layer(in_features, hidden_features, has_bias=bias[0])
        self.act = act_layer
        self.drop1 = nn.Dropout(p=drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, has_bias=bias[1])
        self.drop2 = nn.Dropout(p=drop_probs[1])

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x, approximate="tanh")
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
