import json
import os
import sys
from abc import ABC
from typing import Callable, List, Tuple

import cv2
import numpy as np

sys.path.append("../stable_diffusion_v2/")
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
from ldm.data.transforms import CannyRandomThreshold

from mindspore.dataset.vision import Resize, ToTensor

from mindone.data import BaseDataset


class CondDataset(BaseDataset, ABC):
    """
    Base class for datasets with additional condition (e.g., depth, segmentation, canny, etc.).
    """

    def __init__(self):
        self._data = []
        self.output_columns = ["image", "condition", "caption"]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        image = cv2.cvtColor(cv2.imread(self._data[idx]["image"]), cv2.COLOR_BGR2RGB)
        cond = cv2.cvtColor(cv2.imread(self._data[idx]["condition"]), cv2.COLOR_BGR2RGB)

        return image, cond, self._data[idx]["caption"]

    def __len__(self):
        return len(self._data)


class COCOStuff(CondDataset):
    """
    `COCO-Stuff <https://github.com/nightrome/cocostuff>`__ segmentation dataset. Contains 118K train and 5K validation
    with 80 thing classes, 91 stuff classes and 1 class 'unlabeled'. Each image has 5 captions.

    Grayscale masks have to be colored first with `t2i_tools/cocostuff_colorize_mask.py`.

    Grayscale masks are located in `stuffthingmaps_trainval2017.zip`, and annotations are located in
    `annotations_trainval2017.zip/annotations/captions_train2017.json`

    Args:
        image_dir: path to directory with images.
        masks_path: path to directory with colored masks.
        label_path: path to json file with annotations.
    """

    def __init__(self, image_dir: str, masks_path: str, label_path: str):
        super().__init__()
        with open(label_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)["annotations"]

        self._data = []
        for file in data:
            image_path = os.path.join(image_dir, f"{file['image_id']:012d}.jpg")
            mask_path = os.path.join(masks_path, f"{file['image_id']:012d}.png")
            assert os.path.exists(image_path), f"COCO-Stuff: Image {image_path} does not exist!"
            assert os.path.exists(mask_path), f"COCO-Stuff: Mask {mask_path} does not exist!"

            self._data.append({"image": image_path, "condition": mask_path, "caption": file["caption"]})

    @staticmethod
    def train_transforms(cond: str, tokenizer: Callable[[str], np.ndarray]) -> List[dict]:
        transforms = [
            {"operations": tokenizer, "input_columns": ["caption"]},
            {
                "operations": [Resize((512, 512)), lambda x: (x / 127.5 - 1.0).astype(np.float32)],
                "input_columns": ["image"],
            },
        ]

        if cond.lower() == "canny":  # generate Canny conditions with dynamic thresholds during training
            transforms.append({"operations": CannyRandomThreshold(), "input_columns": ["condition"]})

        transforms.append({"operations": [Resize((512, 512)), ToTensor()], "input_columns": ["condition"]})
        return transforms
