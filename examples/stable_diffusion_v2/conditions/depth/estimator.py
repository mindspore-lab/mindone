from typing import List, Union

import numpy as np
import PIL
from conditions.depth.midas import midas_v3_dpt_large
from PIL import Image

import mindspore as ms
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.amp import auto_mixed_precision

# import logging
# _logger = logging.getLogger(__name__)

__all__ = ["DepthEstimator"]


class DepthEstimator:
    def __init__(
        self,
        model_type="midas_v3_dpt_large_384",
        estimator_ckpt_path="models/depth_estimator/midas_v3_dpt_large-c8fd1049.ckpt",
        amp_level="O2",
    ):
        self.amp_level = amp_level
        dtype = ms.float32 if amp_level == "O0" else ms.float16
        if model_type == "midas_v3_dpt_large_384":
            depth_model = midas_v3_dpt_large(pretrained=True, ckpt_path=estimator_ckpt_path, dtype=dtype)
        else:
            # TODO: support midas v3 hybrid
            raise NotImplementedError

        self.depth_estimator = auto_mixed_precision(depth_model, amp_level=amp_level)

    def preprocess(self, images):
        if not isinstance(images, list):
            images = [images]

        # 1. preproess
        # hyper-params ref: https://huggingface.co/stabilityai/stable-diffusion-2-depth/blob/main/feature_extractor/preprocessor_config.json
        h = w = 384  # input image size for depth estimator
        rescale = True
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        # 1.1 resize to 384
        images = [img.resize((w, h), resample=Image.BICUBIC) for img in images]  # resample=2 => BICUBIC
        images = [np.array(img, dtype=np.float32) for img in images]
        images = np.array(images, dtype=np.float32)  # [bs, h, w, 3]

        # 1.2 rescale to [0, 1]
        if rescale:
            images = images / 255.0
        # 1.3 normalize to [-1, 1]
        images = (images - mean) / std
        # 1.4 format tensor batch [bs, 3, h, w]
        images = np.transpose(images, (0, 3, 1, 2))
        if self.amp_level != "O0":
            images = Tensor(images, dtype=mstype.float32)
        else:
            images = Tensor(images, dtype=mstype.float16)
        assert (
            len(images.shape) == 4 and images.shape[1] == 3
        ), f"Expecting model input shape: [bs, 3, H, W], but got {images.shape}"

        return images

    def __call__(self, images: Union[PIL.Image.Image, List[PIL.Image.Image]]):
        """
        Use MiDas as depth estimator.
        Args:
            images: rgb image as PIL object, shape [h, w, 3], value: 0-255
                or, list of PIL images  [n, h, w, 3]

        return:
            depth map as numpy array, shpae [384, 384]
                or [n, 384, 384]
        """

        images = self.preprocess(images)

        # 2. infer
        depth_maps = self.depth_estimator(images).asnumpy().astype(np.float32)  # [bs, 1, h, w]
        depth_maps = np.squeeze(depth_maps)  # [bs, h, w] or [h, w]

        return depth_maps
