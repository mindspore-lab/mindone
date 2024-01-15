"""
transform methods for vision models
"""
import numpy as np
from PIL import Image

import mindspore as ms
from mindspore.dataset import vision

__all__ = ["BatchResize", "BCHW2BHWC", "BatchPILize", "BatchNormalize", "BatchCenterCrop", "BatchToTensor"]


class BCHW2BHWC:
    """
    Transform a batch of image from CHW to HWC.

    Args:
         image_batch (tensor, numpy.array, PIL.Image, list): for tensor or numpy input, the
         channel should be (bz, c, h, w) or (c, h, w). for list, the item should be
        PIL.Image or numpy.array (c, h, w).

    Return:
         transformed image batch: for numpy or tensor input, return a numpy array, the channel
         is (bz, h, w, c) or (h, w, c); for PIL.Image input, it is returned directly.
    """

    def __call__(self, image_batch):
        """the call function"""
        if isinstance(image_batch, ms.Tensor):
            image_batch = image_batch.asnumpy()

        if isinstance(image_batch, list):
            return [self(item) for item in image_batch]
        if isinstance(image_batch, np.ndarray):
            if len(image_batch.shape) == 4:
                return image_batch.transpose(0, 2, 3, 1)
            if len(image_batch.shape) == 3:
                return image_batch.transpose(1, 2, 0)
            raise ValueError(f"the rank of image_batch should be 3 or 4," f" but got {len(image_batch.shape)}")
        if isinstance(image_batch, Image.Image):
            return image_batch
        raise TypeError(f"the type {type(image_batch)} of image_batch is unsupported.")


class BatchResize:
    """
    Resize a batch of image to the given shape.

    Args:
         image_resolution (int): the target size.
    """

    def __init__(self, image_resolution):
        self.sizer = vision.Resize(image_resolution)

    def __call__(self, image_batch):
        """
        The forward process.

        Args:
            image_batch (tensor, numpy.array, PIL.Image, list): for tensor or numpy input,
            the shape should be (bz, h, w, c) or (h, w, c). for list, the item should be
            PIL.Image or numpy.array (h, w, c).

        Returns:
            resized image batch: for numpy or tensor input, return a numpy array;
            for PIL.Image input, it returns PIL.Image.
        """
        if isinstance(image_batch, ms.Tensor):
            image_batch = image_batch.asnumpy()

        if isinstance(image_batch, list):
            return [self.sizer(item) for item in image_batch]
        if isinstance(image_batch, np.ndarray):
            if len(image_batch.shape) == 4:
                return np.row_stack([self.sizer(item)[np.newaxis, :] for item in image_batch])
            if len(image_batch.shape) == 3:
                return self.sizer(image_batch)
            raise ValueError(f"the rank of image_batch should be 3 or 4," f" but got {len(image_batch.shape)}")
        if isinstance(image_batch, Image.Image):
            return self.sizer(image_batch)
        raise TypeError(f"the type {type(image_batch)} of image_batch is unsupported.")


class BatchCenterCrop:
    """
    CenterCrop a batch of image to the given shape.

    Args:
         image_resolution (int): the target size.
    """

    def __init__(self, image_resolution):
        self.crop = vision.CenterCrop(image_resolution)

    def __call__(self, image_batch):
        """
        The forward process.

        Args:
            image_batch (tensor, numpy.array, PIL.Image, list): for tensor or numpy input,
            the shape should be (bz, h, w, c) or (h, w, c). for list, the item should be
            PIL.Image or numpy.array (h, w, c).

        Returns:
            center cropped image batch: for numpy or tensor input, return a numpy array, the shape
            is (bz, image_resolution, image_resolution, c) or (image_resolution,
            image_resolution, c); for PIL.Image input, it is returned with shape (image_resolution,
            image_resolution).
        """
        if isinstance(image_batch, ms.Tensor):
            image_batch = image_batch.asnumpy()

        if isinstance(image_batch, list):
            return [self.crop(item) for item in image_batch]
        if isinstance(image_batch, np.ndarray):
            if len(image_batch.shape) == 4:
                return np.row_stack([self.crop(item)[np.newaxis, :] for item in image_batch])
            if len(image_batch.shape) == 3:
                return self.crop(image_batch)
            raise ValueError(f"the rank of image_batch should be 3 or 4," f" but got {len(image_batch.shape)}")
        if isinstance(image_batch, Image.Image):
            return self.crop(image_batch)
        raise TypeError(f"the type {type(image_batch)} of image_batch is unsupported.")


class BatchToTensor:
    """Transform a batch of image to tensor and scale to (0, 1)."""

    def __init__(self):
        self.totensor = ms.dataset.vision.ToTensor()

    def __call__(self, image_batch):
        """
        The forward process.

        Args:
            image_batch (tensor, numpy.array, PIL.Image, list): for tensor or numpy input,
            the rank should be 4 or 3. for list, the item should be PIL.Image or numpy.array.

        Returns:
            return a tensor or a list of tensor.
        """
        if isinstance(image_batch, ms.Tensor):
            image_batch = image_batch.asnumpy()

        if isinstance(image_batch, list):
            return [self.totensor(item) for item in image_batch]
        if isinstance(image_batch, np.ndarray):
            if len(image_batch.shape) == 4:
                return np.row_stack([self.totensor(item)[np.newaxis, :] for item in image_batch])
            if len(image_batch.shape) == 3:
                return self.totensor(image_batch)
            raise ValueError(f"the rank of image_batch should be 3 or 4," f" but got {len(image_batch.shape)}")
        if isinstance(image_batch, Image.Image):
            return self.totensor(image_batch)
        raise TypeError(f"the type {type(image_batch)} of image_batch is unsupported.")


class BatchNormalize:
    """Normalize a batch of image."""

    def __init__(
        self, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), is_hwc=False
    ):
        self.normalize = vision.Normalize(mean=mean, std=std, is_hwc=is_hwc)

    def __call__(self, image_batch):
        """
        The forward process.

        Args:
            image_batch (tensor, numpy.array, list): for tensor or numpy input,
            the rank should be 4 or 3. for list, the item should be numpy.array.

        Returns:
            return a tensor or a list of tensor.
        """
        if isinstance(image_batch, ms.Tensor):
            image_batch = image_batch.asnumpy()

        if isinstance(image_batch, list):
            return [self.normalize(item) for item in image_batch]
        if isinstance(image_batch, np.ndarray):
            if len(image_batch.shape) == 3:
                return self.normalize(image_batch)
            if len(image_batch.shape) == 4:
                return np.row_stack([self.normalize(item)[np.newaxis, :] for item in image_batch])
            raise ValueError(f"the rank of image_batch should be 3 or 4," f" but got {len(image_batch.shape)}")
        raise TypeError(f"the type {type(image_batch)} of image_batch is unsupported.")


class BatchPILize:
    """transform a batch of image to PIL.Image list."""

    def __call__(self, image_batch):
        """
        The forward process.

        Args:
            image_batch (tensor, numpy.array, list): for tensor or numpy input,
            the rank should be 4 or 3. for list, the item should be PIL.Image.

        Returns:
            return a tensor or a list of tensor.
        """
        if isinstance(image_batch, Image.Image):
            return image_batch

        if isinstance(image_batch, list):
            for item in image_batch:
                if not isinstance(item, Image.Image):
                    raise TypeError(
                        "unsupported type in list,"
                        " when the image_batch is a list,"
                        " the item in list should be PIL.Image."
                    )
            return image_batch

        if isinstance(image_batch, ms.Tensor):
            image_batch = image_batch.asnumpy()

        if isinstance(image_batch, np.ndarray):
            if len(image_batch.shape) == 4:
                return [Image.fromarray(item.astype(np.uint8)) for item in image_batch]
            if len(image_batch.shape) == 3:
                return Image.fromarray(image_batch.astype(np.uint8))
            raise ValueError(f"the rank of image_batch should be 3 or 4," f" but got {len(image_batch.shape)}")

        raise ValueError("unsupported input type.")
