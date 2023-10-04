from typing import List, Optional, Union

import numpy as np
from PIL import Image

from ..utils import get_abspath_of_weights
from .clip_utils import (
    BCHW2BHWC,
    BatchCenterCrop,
    BatchNormalize,
    BatchPILize,
    BatchResize,
    BatchToTensor,
    CLIPTokenizer,
)

__all__ = ["CLIPTextProcessor", "CLIPImageProcessor"]


def load_ckpt_tokenizer(tokenizer_path):
    text_processor = CLIPTokenizer(tokenizer_path, pad_token="!")
    return text_processor


class CLIPImageProcessor:
    """
    CLIPImageProcessor.

    Args:
        image_resolution (int): The target size.
    """

    def __init__(self, image_resolution: Optional[int] = 224):
        self.config = {"image_resolution": image_resolution}
        self.bchw2bhwc = BCHW2BHWC()
        self.batch_pilizer = BatchPILize()
        self.batch_resizer = BatchResize(image_resolution)
        self.batch_crop = BatchCenterCrop(image_resolution)
        self.batch_totensor = BatchToTensor()
        self.batch_normalizer = BatchNormalize()

    def __call__(self, image_data, **kwargs):
        """forward process"""
        return self.preprocess(image_data, **kwargs)

    def preprocess(self, images: Union[Image.Image, np.ndarray, List[Image.Image]]) -> np.ndarray:
        r"""
        Preprocess Required By Base Processor.

        Args:
            images (ms.Tensor, PIL.Image, numpy.array, List[PIL.Image]): A batch of images.

        Return:
            A 4-rank tensor for a batch of images.
        """
        if not self._bhwc_check(images):
            images = self.bchw2bhwc(images)
        images = self.batch_pilizer(images)
        images = self.batch_resizer(images)
        images = self.batch_crop(images)
        images = self.batch_totensor(images)
        images = self.batch_normalizer(images)

        if isinstance(images, list):
            return np.row_stack([np.expand_dims(item, axis=0) for item in images])
        if len(images.shape) == 4:
            return images
        return np.expand_dims(images, axis=0)

    def _bhwc_check(self, image_batch: Union[Image.Image, np.ndarray, List[Image.Image]]):
        r"""Bhwc_check"""
        if isinstance(image_batch, np.ndarray):
            if image_batch.shape[-1] == 3:
                return True
        if isinstance(image_batch, (list, Image.Image)):
            return True
        return False


class CLIPTextProcessor:
    def __init__(self, path: str, return_tensor: bool = False) -> None:
        self.tokenizer = load_ckpt_tokenizer(get_abspath_of_weights(path))
        self.return_tensor = return_tensor

    def __call__(self, text_prompt: List[str]) -> np.ndarray:
        output = self.tokenizer(text_prompt, padding="max_length", max_length=77)["input_ids"]
        output = np.array(output, dtype=np.int32)
        return output
