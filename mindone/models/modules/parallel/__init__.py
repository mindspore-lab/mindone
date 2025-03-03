from mindspore import mint, nn

from .conv import Conv1d, Conv2d, Conv3d, Mint_Conv2d, Mint_Conv3d
from .dense import Dense, Linear

# {Original MindSpore Cell: New Cell in ZeRO3}
PARALLEL_MODULES = {
    nn.Conv1d: Conv1d,
    nn.Conv2d: Conv2d,
    nn.Conv3d: Conv3d,
    nn.Dense: Dense,
}

try:
    from mindspore.mint.nn import Conv2d

    PARALLEL_MODULES[mint.nn.Conv2d] = Mint_Conv2d
except ImportError:
    pass

try:
    from mindspore.mint.nn import Conv3d

    PARALLEL_MODULES[mint.nn.Conv3d] = Mint_Conv3d
except ImportError:
    pass

try:
    from mindspore.mint.nn import Linear as m_Linear

    PARALLEL_MODULES[mint.nn.Linear] = Linear
except ImportError:
    pass


__all__ = ["Conv1d", "Conv2d", "Conv3d", "Mint_Conv2d", "Mint_Conv3d", "Dense", "Linear"]
