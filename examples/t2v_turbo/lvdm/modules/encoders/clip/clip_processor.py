"""
CLIPProcessor
"""
from typing import List, Optional, Union

import numpy as np
import PIL

import mindspore as ms

from .utils import BCHW2BHWC, BatchCenterCrop, BatchNormalize, BatchPILize, BatchResize, BatchToTensor


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

    def preprocess(self, images: Union[ms.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]], **kwargs):
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

        kwargs.pop("other", None)
        if isinstance(images, list):
            return ms.Tensor(np.row_stack([np.expand_dims(item, axis=0) for item in images]))
        if len(images.shape) == 4:
            return ms.Tensor(images)
        return ms.Tensor(np.expand_dims(images, axis=0))

    def _bhwc_check(self, image_batch: Union[ms.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]]):
        r"""Bhwc_check"""
        if isinstance(image_batch, np.ndarray):
            if image_batch.shape[-1] == 3:
                return True
        if isinstance(image_batch, ms.Tensor):
            if image_batch.asnumpy().shape[-1] == 3:
                return True
        if isinstance(image_batch, (list, PIL.Image.Image)):
            return True
        return False
