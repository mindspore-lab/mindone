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


import json
from abc import ABC, abstractmethod

import numpy as np
from megfile import smart_exists, smart_open, smart_path_join
from PIL import Image

import mindspore as ms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter


class BaseDataset(ABC):
    def __init__(self, root_dirs: list[str], meta_path: str):
        super().__init__()
        self.root_dirs = root_dirs
        self.uids = self._load_uids(meta_path)

    def __len__(self):
        return len(self.uids)

    @abstractmethod
    def inner_get_item(self, idx):
        pass

    def __getitem__(self, idx):
        return self.inner_get_item(idx)

    @staticmethod
    def _load_uids(meta_path: str):
        # meta_path is a json file
        with open(meta_path, "r") as f:
            uids = json.load(f)
        return uids

    @staticmethod
    def _load_rgba_image(file_path, bg_color: float = 1.0, resize=None, crop_pos=None, crop_size=None):
        """Load and blend RGBA image to RGB with certain background, 0-1 scaled
        Transform image properly (resize, and crop):
            - resize: int
            - crop_pos: [int, int]
            - crop_size: int
        """
        # read image
        rgba = np.array(Image.open(smart_open(file_path, "rb")))

        # image transformation
        if resize is not None:
            rgba = vision.Resize([resize, resize], Inter.BICUBIC)(rgba)
        if (crop_pos is not None) and (crop_size is not None):  # rand crop
            assert (crop_pos[0] + crop_size <= rgba.shape[0]) and (crop_pos[1] + crop_size <= rgba.shape[1])
            rgba = vision.Crop(crop_pos, crop_size)(rgba)

        # convert to Tensor, in shape [B, C, H, W]
        rgba = ms.Tensor(rgba).float() / 255.0
        rgba = rgba.permute((2, 0, 1)).unsqueeze(0)
        rgb = rgba[:, :3, :, :] * rgba[:, 3:4, :, :] + bg_color * (1 - rgba[:, 3:, :, :])

        return rgb

    @staticmethod
    def _locate_datadir(root_dirs, uid, locator: str):
        for root_dir in root_dirs:
            datadir = smart_path_join(root_dir, uid, locator)
            if smart_exists(datadir):
                return root_dir
        raise FileNotFoundError(f"Cannot find valid data directory for uid {uid}")
