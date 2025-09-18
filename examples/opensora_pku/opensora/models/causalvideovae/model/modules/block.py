# Adapted from
# https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/model/modules/block.py

from mindspore import nn


class Block(nn.Cell):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
