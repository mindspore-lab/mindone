import re

import mindspore.nn as nn


class IdentityMap(nn.Cell):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Cell):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.SequentialCell(
            nn.Dense(channels, channels),
            # cong TODO: according to ms document,
            # set approximate to False can get similar result with pt
            # need to check it
            nn.GELU(approximate=False),
            nn.Dense(channels, channels),
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = config.get("mm_projector_type", "linear")

    if projector_type == "linear":
        # return nn.Dense(config.mm_hidden_size, config.hidden_size)
        return nn.Dense(config["mm_hidden_size"], config["hidden_size"])

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Dense(config["mm_hidden_size"], config["hidden_size"])]
        for _ in range(1, mlp_depth):
            # cong TODO: according to ms document,
            # set approximate to False can get similar result with pt
            # need to check it
            modules.append(nn.GELU(approximate=False))
            modules.append(nn.Dense(config["hidden_size"], config["hidden_size"]))
        return nn.SequentialCell(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
