# Copyright 2024 Open AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import mindspore as ms
from mindspore import ops


@dataclass
class DifferentiableProjectiveCamera:
    """
    Implements a batch, differentiable, standard pinhole camera
    """

    origin: ms.Tensor  # [batch_size x 3]
    x: ms.Tensor  # [batch_size x 3]
    y: ms.Tensor  # [batch_size x 3]
    z: ms.Tensor  # [batch_size x 3]
    width: int
    height: int
    x_fov: float
    y_fov: float
    shape: Tuple[int]

    def __post_init__(self):
        assert self.x.shape[0] == self.y.shape[0] == self.z.shape[0] == self.origin.shape[0]
        assert self.x.shape[1] == self.y.shape[1] == self.z.shape[1] == self.origin.shape[1] == 3
        assert len(self.x.shape) == len(self.y.shape) == len(self.z.shape) == len(self.origin.shape) == 2

    def resolution(self):
        return ms.Tensor.from_numpy(np.array([self.width, self.height], dtype=np.float32))

    def fov(self):
        return ms.Tensor.from_numpy(np.array([self.x_fov, self.y_fov], dtype=np.float32))

    def get_image_coords(self) -> ms.Tensor:
        """
        :return: coords of shape (width * height, 2)
        """
        pixel_indices = ops.arange(self.height * self.width).to(ms.int32)
        coords = ops.stack(
            [
                pixel_indices % self.width,
                ops.div(pixel_indices, self.width, rounding_mode="trunc"),
            ],
            axis=1,
        )
        return coords

    @property
    def camera_rays(self):
        batch_size, *inner_shape = self.shape
        inner_batch_size = int(np.prod(inner_shape))

        coords = self.get_image_coords()
        coords = ops.broadcast_to(coords.unsqueeze(0), (batch_size * inner_batch_size, *coords.shape))
        rays = self.get_camera_rays(coords)

        rays = rays.view(batch_size, inner_batch_size * self.height * self.width, 2, 3)

        return rays

    def get_camera_rays(self, coords: ms.Tensor) -> ms.Tensor:
        batch_size, *shape, n_coords = coords.shape
        assert n_coords == 2
        assert batch_size == self.origin.shape[0]

        flat = coords.view(batch_size, -1, 2)

        res = self.resolution()
        fov = self.fov()

        fracs = (flat.float() / (res - 1)) * 2 - 1
        fracs = fracs * ops.tan(fov / 2)

        fracs = fracs.view(batch_size, -1, 2)
        directions = (
            self.z.view(batch_size, 1, 3)
            + self.x.view(batch_size, 1, 3) * fracs[:, :, :1]
            + self.y.view(batch_size, 1, 3) * fracs[:, :, 1:]
        )
        directions = directions / directions.norm(dim=-1, keepdim=True)
        rays = ops.stack(
            [
                ops.broadcast_to(self.origin.view(batch_size, 1, 3), (batch_size, directions.shape[1], 3)),
                directions,
            ],
            axis=2,
        )
        return rays.view(batch_size, *shape, 2, 3)

    def resize_image(self, width: int, height: int) -> "DifferentiableProjectiveCamera":
        """
        Creates a new camera for the resized view assuming the aspect ratio does not change.
        """
        assert width * self.height == height * self.width, "The aspect ratio should not change."
        return DifferentiableProjectiveCamera(
            origin=self.origin,
            x=self.x,
            y=self.y,
            z=self.z,
            width=width,
            height=height,
            x_fov=self.x_fov,
            y_fov=self.y_fov,
        )


def create_pan_cameras(size: int) -> DifferentiableProjectiveCamera:
    origins = []
    xs = []
    ys = []
    zs = []
    for theta in np.linspace(0, 2 * np.pi, num=20):
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        z /= np.sqrt(np.sum(z**2))
        origin = -z * 4
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        y = np.cross(z, x)
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return DifferentiableProjectiveCamera(
        origin=ms.Tensor.from_numpy(np.stack(origins, axis=0)).float(),
        x=ms.Tensor.from_numpy(np.stack(xs, axis=0)).float(),
        y=ms.Tensor.from_numpy(np.stack(ys, axis=0)).float(),
        z=ms.Tensor.from_numpy(np.stack(zs, axis=0)).float(),
        width=size,
        height=size,
        x_fov=0.7,
        y_fov=0.7,
        shape=(1, len(xs)),
    )
