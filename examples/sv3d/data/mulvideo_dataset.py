import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from PIL import Image

from mindspore.dataset.vision import Resize

sys.path.append("../../")  # FIXME: remove in future when mindone is ready for install
from mindone.data import BaseDataset

_logger = logging.getLogger("")


def read_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_im(path, color):
    """replace background pixel with random color in rendering"""
    pil_img = Image.open(path)  # h w c
    image = np.asarray(pil_img, dtype=np.float32) / 255.0
    alpha = image[:, :, 3:]
    image = image[:, :, :3] * alpha + color * (1 - alpha)
    return image


class MulviewVideoDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        metadata: str,
        image_dir: str,
        frames: int,
    ):
        self.root_dir = Path(root_dir)
        self.paths = read_pickle(os.path.join(root_dir, metadata))
        self.image_dir = image_dir
        self._frames = frames
        self._bg_white = [1.0, 1.0, 1.0]
        self.output_columns = ["frames", "cond_aug"]

        _logger.info("============= length of dataset %d =============" % len(self.paths))

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        input_image_path = os.path.join(self.root_dir, self.image_dir, self.paths[idx])
        frames = []
        for img_idx in range(self._frames):
            img = load_im(
                os.path.join(input_image_path, "%03d.png" % (img_idx + 8)), self._bg_white
            )  # to select the correct frame to start training
            frames.append(img)

        noise_strength = np.random.lognormal(-3.0, 0.5**2)

        return np.stack(frames), noise_strength

    def __len__(self):
        return len(self.paths)

    def train_transforms(self) -> List[dict]:
        return [
            {
                "operations": [
                    Resize(576),
                    # CenterCrop((576, 1024)),
                    lambda x: np.transpose(x, (0, 3, 1, 2)).astype(np.float32),  # ms.HWC2CHW() doesn't support 4D data
                    # lambda x: (x / 127.5 - 1.0).astype(np.float32),
                ],
                "input_columns": ["frames"],
            },
            {
                "operations": [
                    lambda frames, aug: (
                        frames,
                        frames[0],
                        frames[0] + aug * np.random.randn(*frames[0].shape).astype(np.float32),
                        np.tile(aug, (self._frames, 1)).astype(np.float32),
                    )
                ],
                "input_columns": ["frames", "cond_aug"],
                "output_columns": ["frames", "cond_frames_without_noise", "cond_frames", "cond_aug"],
            },
        ]


class SamplingImageDataset(BaseDataset):
    def __init__(self, image_path):
        self.image_path = [image_path]

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        img = load_im(self.image_path[idx])
        return np.array(img)
