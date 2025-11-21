# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Fast Image processor class for DINOv3."""

from typing import Optional, Union

from ...image_processing_base import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, group_images_by_shape, reorder_images
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling, SizeDict
from ...utils import (
    TensorType,
    logging,
)

import mindspore as ms
from mindspore import mint

logger = logging.get_logger(__name__)


class DINOv3ViTImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 224, "width": 224}
    do_resize = True
    do_rescale = True
    do_normalize = True

    # Overridden for DINOv3 to preserve order of transforms
    # rescale -> resize -> normalize
    def _preprocess(
        self,
        images: list["ms.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_rescale:
                stacked_images = self.rescale(stacked_images, rescale_factor)
            if do_resize:
                # TODO mindspore.dataset.vision.Resize could only support (H, W, 3) format,
                #  batch_size stacked image should be computed in one iteration
                # batch_size, channels = stacked_images.shape[0], stacked_images.shape[1]
                # stacked_images_updated = mint.zeros((batch_size, channels, resized_height, resized_width), dtype=stacked_images.dtype)
                stacked_images_updated = []
                for i in range(len(stacked_images)):
                    stacked_images_updated.append(
                        self.resize(
                            image=stacked_images[i], size=size, interpolation=interpolation, antialias=True
                        )
                    )
                stacked_images = stacked_images_updated
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            if do_normalize:
                stacked_images = self.normalize(stacked_images, image_mean, image_std)
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = mint.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["DINOv3ViTImageProcessorFast"]
