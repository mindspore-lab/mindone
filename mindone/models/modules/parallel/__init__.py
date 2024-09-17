from mindspore import nn

from .conv import Conv1d, Conv2d, Conv3d
from .dense import Dense

# {Original MindSpore Cell: New Cell in ZeRO3}
PARALLEL_MODULES = {
    nn.Conv1d: Conv1d,
    nn.Conv2d: Conv2d,
    nn.Conv3d: Conv3d,
    nn.Dense: Dense,
}
__all__ = ["Conv1d", "Conv2d", "Conv3d", "Dense"]
