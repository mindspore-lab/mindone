from mindspore import mint, nn

from .conv import Conv1d, Conv2d, Conv3d, Mint_Conv2d, Mint_Conv3d
from .dense import Dense, Linear
from .moe_text_experts import MoeTextExperts

# {Original MindSpore Cell: New Cell in ZeRO3}
PARALLEL_MODULES = {
    nn.Conv1d: Conv1d,
    nn.Conv2d: Conv2d,
    nn.Conv3d: Conv3d,
    nn.Dense: Dense,
    mint.nn.Conv2d: Mint_Conv2d,
    mint.nn.Conv3d: Mint_Conv3d,
    mint.nn.Linear: Linear,
}

SPECIAL_CASE_FOR_PARALLEL_MODULES = {nn.Cell: MoeTextExperts}

__all__ = ["Conv1d", "Conv2d", "Conv3d", "Mint_Conv2d", "Mint_Conv3d", "Dense", "Linear"]
