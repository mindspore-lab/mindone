# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import random
from typing import Union

import numpy as np
from megfile import smart_open, smart_path_join

import mindspore as ms
from mindspore import mint

from ..utils.proxy import no_proxy
from .base import BaseDataset
from .cam_utils import build_camera_principle, build_camera_standard, camera_normalization_objaverse

__all__ = ["ObjaverseDataset"]


class ObjaverseDataset(BaseDataset):
    def __init__(
        self,
        root_dirs: list[str],
        meta_path: str,
        sample_side_views: int,
        render_image_res_low: int,
        render_image_res_high: int,
        render_region_size: int,
        source_image_res: int,
        normalize_camera: bool,
        normed_dist_to_center: Union[float, str] = None,
        num_all_views: int = 32,
    ):
        super().__init__(root_dirs, meta_path)
        self.sample_side_views = sample_side_views
        self.render_image_res_low = render_image_res_low
        self.render_image_res_high = render_image_res_high
        self.render_region_size = render_region_size
        self.source_image_res = source_image_res
        self.normalize_camera = normalize_camera
        self.normed_dist_to_center = normed_dist_to_center
        self.num_all_views = num_all_views

    @staticmethod
    def _load_pose(file_path):
        pose = np.load(smart_open(file_path, "rb"))
        pose = ms.Tensor(pose).float()  # C2W [R|t]: matrix 3x4
        return pose

    @no_proxy
    def inner_get_item(self, idx):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """
        uid = self.uids[idx]
        root_dir = self._locate_datadir(self.root_dirs, uid, locator="intrinsics.npy")

        pose_dir = os.path.join(root_dir, uid, "pose")
        rgba_dir = os.path.join(root_dir, uid, "rgba")
        intrinsics_path = os.path.join(root_dir, uid, "intrinsics.npy")

        # load intrinsics
        intrinsics = np.load(smart_open(intrinsics_path, "rb"))
        intrinsics = ms.Tensor(intrinsics, dtype=ms.float32)

        # sample views (incl. source view and side views)
        sample_views = np.random.choice(range(self.num_all_views), self.sample_side_views + 1, replace=False)
        poses, rgbs, bg_colors = [], [], []
        source_image = None
        render_image_res = np.random.randint(self.render_image_res_low, self.render_image_res_high + 1)
        # intended crop region. NOTE: ops.randint will after some interations encounters "RuntimeError: SyncStream failed for op aclnnCast"
        anchors = ms.numpy.randint(0, render_image_res - self.render_region_size + 1, (self.sample_side_views + 1, 2))
        for idx, view in enumerate(sample_views):
            pose_path = smart_path_join(pose_dir, f"{view:03d}.npy")
            rgba_path = smart_path_join(rgba_dir, f"{view:03d}.png")
            pose = self._load_pose(pose_path)
            bg_color = random.choice([0.0, 0.5, 1.0])

            # adjust render image resolution and sample intended rendering region
            crop_pos = [anchors[idx, 0].item(), anchors[idx, 1].item()]
            rgb = self._load_rgba_image(
                rgba_path,
                bg_color=bg_color,
                resize=render_image_res,
                crop_pos=crop_pos,
                crop_size=self.render_region_size,
            )

            poses.append(pose)
            rgbs.append(rgb)
            bg_colors.append(bg_color)

            if source_image is None:
                # load source image and adjust resolution
                source_image = self._load_rgba_image(
                    rgba_path, bg_color=1.0, resize=self.source_image_res, crop_pos=None, crop_size=None
                )

        assert source_image is not None, "Really bad luck!"
        poses = mint.stack(poses, dim=0)
        rgbs = mint.cat(rgbs, dim=0)
        source_image = source_image.squeeze(0)  # [1, C, H, W] -> [C, H, W]

        if self.normalize_camera:
            poses = camera_normalization_objaverse(self.normed_dist_to_center, poses)

        # build source and target camera features
        source_camera = build_camera_principle(poses[:1], intrinsics.unsqueeze(0)).squeeze(0)  # [12+4, ]
        render_camera = build_camera_standard(poses, intrinsics.tile((poses.shape[0], 1, 1)))  # [N, 16+9]

        # image value in [0, 1]
        source_image = mint.clamp(source_image, 0.0, 1.0)  # [C, H, W]
        cropped_render_image = mint.clamp(rgbs, 0.0, 1.0)  # [side+1, C, H, W]

        return (
            source_camera,
            render_camera,
            source_image,
            cropped_render_image,
            anchors,
            ms.Tensor([[render_image_res]], dtype=ms.float32).tile((self.sample_side_views + 1, 1)),
            ms.Tensor(bg_colors, dtype=ms.float32).unsqueeze(-1),
        )
