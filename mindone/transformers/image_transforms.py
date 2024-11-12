import warnings
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

import PIL

import mindspore
from mindspore import ops

from .image_utils import (
    ChannelDimension,
    ImageInput,
    get_channel_dimension_axis,
    get_image_size,
    infer_channel_dimension_format,
)

def to_channel_dimension_format(
    image: np.ndarray,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> np.ndarray:
    """
    Converts `image` to the channel dimension format specified by `channel_dim`.

    Args:
        image (`numpy.ndarray`):
            The image to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.
        input_channel_dim (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `np.ndarray`: The image with the channel dimension set to `channel_dim`.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        image = image.transpose((2, 0, 1))
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.transpose((1, 2, 0))
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))

    return image

def _rescale_for_pil_conversion(image):
    """
    Detects whether or not the image needs to be rescaled before being converted to a PIL image.

    The assumption is that if the image is of type `np.float` and all values are between 0 and 1, it needs to be
    rescaled.
    """
    if image.dtype == np.uint8:
        do_rescale = False
    elif np.allclose(image, image.astype(int)):
        if np.all(0 <= image) and np.all(image <= 255):
            do_rescale = False
        else:
            raise ValueError(
                "The image to be converted to a PIL image contains values outside the range [0, 255], "
                f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
            )
    elif np.all(0 <= image) and np.all(image <= 1):
        do_rescale = True
    else:
        raise ValueError(
            "The image to be converted to a PIL image contains values outside the range [0, 1], "
            f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
        )
    return do_rescale


def to_pil_image(
    image: Union[np.ndarray, "PIL.Image.Image", "mindspore.Tensor"],
    do_rescale: Optional[bool] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> "PIL.Image.Image":
    """
    Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
    needed.

    Args:
        image (`PIL.Image.Image` or `numpy.ndarray` or `mindspore.Tensor` or `tf.Tensor`):
            The image to convert to the `PIL.Image` format.
        do_rescale (`bool`, *optional*):
            Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will default
            to `True` if the image type is a floating type and casting to `int` would result in a loss of precision,
            and `False` otherwise.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `PIL.Image.Image`: The converted image.
    """

    if isinstance(image, PIL.Image.Image):
        return image

    # Convert all tensors to numpy arrays before converting to PIL image
    if isinstance(image, mindspore.Tensor):
        image = image.asnumpy()
    elif not isinstance(image, np.ndarray):
        raise ValueError("Input image type not supported: {}".format(type(image)))

    # If the channel has been moved to first dim, we put it back at the end.
    image = to_channel_dimension_format(image, ChannelDimension.LAST, input_data_format)

    # If there is a single channel, we squeeze it, as otherwise PIL can't handle it.
    image = np.squeeze(image, axis=-1) if image.shape[-1] == 1 else image

    # PIL.Image can only store uint8 values so we rescale the image to be between 0 and 255 if needed.
    do_rescale = _rescale_for_pil_conversion(image) if do_rescale is None else do_rescale

    if do_rescale:
        image = rescale(image, 255)

    image = image.astype(np.uint8)
    return PIL.Image.fromarray(image)

def center_crop(
    image: np.ndarray,
    size: Tuple[int, int],
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    return_numpy: Optional[bool] = None,
) -> np.ndarray:
    """
    Crops the `image` to the specified `size` using a center crop. Note that if the image is too small to be cropped to
    the size given, it will be padded (so the returned result will always be of size `size`).

    Args:
        image (`np.ndarray`):
            The image to crop.
        size (`Tuple[int, int]`):
            The target size for the cropped image.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.
        return_numpy (`bool`, *optional*):
            Whether or not to return the cropped image as a numpy array. Used for backwards compatibility with the
            previous ImageFeatureExtractionMixin method.
                - Unset: will return the same type as the input image.
                - `True`: will return a numpy array.
                - `False`: will return a `PIL.Image.Image` object.
    Returns:
        `np.ndarray`: The cropped image.
    """

    if return_numpy is not None:
        warnings.warn("return_numpy is deprecated and will be removed in v.4.33", FutureWarning)

    return_numpy = True if return_numpy is None else return_numpy

    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    if not isinstance(size, Iterable) or len(size) != 2:
        raise ValueError("size must have 2 elements representing the height and width of the output image")

    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    output_data_format = data_format if data_format is not None else input_data_format

    # We perform the crop in (C, H, W) format and then convert to the output format
    image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

    orig_height, orig_width = get_image_size(image, ChannelDimension.FIRST)
    crop_height, crop_width = size
    crop_height, crop_width = int(crop_height), int(crop_width)

    # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (orig_width - crop_width) // 2
    right = left + crop_width

    # Check if cropped area is within image boundaries
    if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
        image = image[..., top:bottom, left:right]
        image = to_channel_dimension_format(image, output_data_format, ChannelDimension.FIRST)
        return image

    # Otherwise, we may need to pad if the image is too small. Oh joy...
    new_height = max(crop_height, orig_height)
    new_width = max(crop_width, orig_width)
    new_shape = image.shape[:-2] + (new_height, new_width)
    new_image = np.zeros_like(image, shape=new_shape)

    # If the image is too small, pad it with zeros
    top_pad = (new_height - orig_height) // 2
    bottom_pad = top_pad + orig_height
    left_pad = (new_width - orig_width) // 2
    right_pad = left_pad + orig_width
    new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

    top += top_pad
    bottom += top_pad
    left += left_pad
    right += left_pad

    new_image = new_image[..., max(0, top) : min(new_height, bottom), max(0, left) : min(new_width, right)]
    new_image = to_channel_dimension_format(new_image, output_data_format, ChannelDimension.FIRST)

    if not return_numpy:
        new_image = to_pil_image(new_image)

    return new_image

def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

    image = (image - mean) / std

    Args:
        image (`np.ndarray`):
            The image to normalize.
        mean (`float` or `Iterable[float]`):
            The mean to use for normalization.
        std (`float` or `Iterable[float]`):
            The standard deviation to use for normalization.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")

    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    channel_axis = get_channel_dimension_axis(image, input_data_format=input_data_format)
    num_channels = image.shape[channel_axis]

    # We cast to float32 to avoid errors that can occur when subtracting uint8 values.
    # We preserve the original dtype if it is a float type to prevent upcasting float16.
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

    if isinstance(mean, Iterable):
        if len(mean) != num_channels:
            raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
    else:
        mean = [mean] * num_channels
    mean = np.array(mean, dtype=image.dtype)

    if isinstance(std, Iterable):
        if len(std) != num_channels:
            raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
    else:
        std = [std] * num_channels
    std = np.array(std, dtype=image.dtype)

    if input_data_format == ChannelDimension.LAST:
        image = (image - mean) / std
    else:
        image = ((image.T - mean) / std).T

    image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
    return image

def rescale(
    image: np.ndarray,
    scale: float,
    data_format: Optional[ChannelDimension] = None,
    dtype: np.dtype = np.float32,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Rescales `image` by `scale`.

    Args:
        image (`np.ndarray`):
            The image to rescale.
        scale (`float`):
            The scale to use for rescaling the image.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the image. If not provided, it will be the same as the input image.
        dtype (`np.dtype`, *optional*, defaults to `np.float32`):
            The dtype of the output image. Defaults to `np.float32`. Used for backwards compatibility with feature
            extractors.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `np.ndarray`: The rescaled image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    rescaled_image = image * scale
    if data_format is not None:
        rescaled_image = to_channel_dimension_format(rescaled_image, data_format, input_data_format)

    rescaled_image = rescaled_image.astype(dtype)

    return rescaled_image