import os

import mindspore as ms
import mindspore.ops as ops
from mindspore import JitConfig, Tensor

try:
    _config = JitConfig(jit_level="O3") if os.environ.get("MS_ENABLE_GE", 0) else None
except ValueError:  # for MS > 2.1
    _config = JitConfig(jit_level="O2")


@ms.jit(jit_config=_config)
def swap_c_t_and_tile(x: Tensor) -> Tensor:
    """Swap the second and third dimension, and duplicated along the first dimension for
    classifier-free guidance
    """
    x = ops.transpose(x, (0, 2, 1, 3, 4))
    x = ops.tile(x, (2, 1, 1, 1, 1))
    return x


@ms.jit(jit_config=_config)
def make_masked_images(imgs: Tensor, masks: Tensor) -> Tensor:
    """Making masked image for condition"""
    imgs = (imgs - 0.5) / 0.5
    masked_imgs = ops.concat([imgs * (1 - masks), (1 - masks)], axis=2)
    return masked_imgs
