import numpy as np

import mindspore as ms
from mindspore import ops

from . import Camera


# the ndc projection
def projection(x=0.1, n=1.0, f=50.0, near_plane=None):
    if near_plane is None:
        near_plane = n
    return np.array(
        [
            [n / x, 0, 0, 0],
            [0, n / -x, 0, 0],
            [0, 0, -(f + near_plane) / (f - near_plane), -(2 * f * near_plane) / (f - near_plane)],
            [0, 0, -1, 0],
        ]
    ).astype(np.float32)


class PerspectiveCamera(Camera):
    def __init__(self, fovy=49.0, device="cuda"):
        super(PerspectiveCamera, self).__init__()
        self.device = device
        focal = np.tan(fovy / 180.0 * np.pi * 0.5)
        self.proj_mtx = (
            ms.Tensor.from_numpy(projection(x=focal, f=1000.0, n=1.0, near_plane=0.1)).to(self.device).unsqueeze(dim=0)
        )

    def project(self, points_bxnx4):
        out = ops.matmul(points_bxnx4, ops.transpose(self.proj_mtx, 1, 2))
        return out
