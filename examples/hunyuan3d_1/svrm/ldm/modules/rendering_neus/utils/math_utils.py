# MIT License

# Copyright (c) 2022 Petr Kellnhofer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import mindspore as ms
from mindspore import mint, ops


def transform_vectors(matrix: ms.Tensor, vectors4: ms.Tensor) -> ms.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = mint.matmul(vectors4, matrix.T)
    return res


def normalize_vecs(vectors: ms.Tensor) -> ms.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (ops.norm(vectors, dim=-1, keepdim=True))


def mindspore_dot(x: ms.Tensor, y: ms.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)


def get_ray_limits_box(rays_o: ms.Tensor, rays_d: ms.Tensor, box_side_length):
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """
    o_shape = rays_o.shape
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    bb_min = [-1 * (box_side_length / 2), -1 * (box_side_length / 2), -1 * (box_side_length / 2)]
    bb_max = [1 * (box_side_length / 2), 1 * (box_side_length / 2), 1 * (box_side_length / 2)]
    bounds = ms.Tensor([bb_min, bb_max], dtype=rays_o.dtype)
    is_valid = mint.ones(rays_o.shape[:-1], dtype=ms.bool_)

    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()

    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]

    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]
    tymax = (bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]

    # Resolve parallel rays.
    is_valid[mint.logical_or(tmin > tymax, tymin > tmax)] = False

    # Use the shortest intersection.
    tmin = mint.max(tmin, tymin)
    tmax = mint.min(tmax, tymax)

    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]
    tzmax = (bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]

    # Resolve parallel rays.
    is_valid[mint.logical_or(tmin > tzmax, tzmin > tmax)] = False

    # Use the shortest intersection.
    tmin = mint.max(tmin, tzmin)
    tmax = mint.min(tmax, tzmax)

    # Mark invalid.
    tmin[mint.logical_not(is_valid)] = -1
    tmax[mint.logical_not(is_valid)] = -2

    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)


def linspace(start: ms.Tensor, stop: ms.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in Mindspore.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = mint.arange(num, dtype=ms.float32) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but script
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out
