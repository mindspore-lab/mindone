import numpy as np

from mindspore import ops


def tensor_to_video(x):
    x = ops.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(1, 0, 2, 3).float().asnumpy()  # c t h w ->
    x = (255 * x).astype(np.uint8)
    return x
