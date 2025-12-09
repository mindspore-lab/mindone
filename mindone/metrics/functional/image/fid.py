import logging
from typing import List, Union

import numpy as np
from PIL import Image

import mindspore
import mindspore.ops as ops
from mindspore import Tensor

from mindone.metrics.utils.feature_extractor_inceptionv3 import (
    FeatureExtractorInceptionV3 as _FeatureExtractorInceptionV3,
)

logger = logging.getLogger(__name__)


def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    r"""Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    a = ops.sum(ops.square(mu1 - mu2), dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = ops.eigvals(sigma1 @ sigma2)
    c = ops.sum(ops.real(ops.sqrt(c)), dim=-1)

    return a + b - 2 * c


def _get_feature_extractor(feature):
    inception = _FeatureExtractorInceptionV3(name="inecption-v3-compat", request_feature=str(feature))
    return inception


def _update_images(images, inception, normalize=False):
    images = (images * 255).to(mindspore.uint8) if normalize else images
    features = inception(images)
    features = features.to(mindspore.double)
    if features.dim() == 1:
        features = features.unsqueeze(0)
    sum_result = ops.sum(features, dim=0)
    cov_sum = ops.mm(ops.transpose(features, (1, 0)), features)
    num_samples = images.shape[0]
    return sum_result, cov_sum, num_samples


def _generate_origin_states(num_features: int):
    mx_num_feats = (num_features, num_features)
    sum_result = ops.zeros(num_features).to(mindspore.double)
    cov_sum = ops.zeros(mx_num_feats).to(mindspore.double)
    num_samples = mindspore.tensor(0).long()
    return sum_result, cov_sum, num_samples


def _compute(
    real_features_sum,
    real_features_cov_sum,
    real_features_num_samples,
    fake_features_sum,
    fake_features_cov_sum,
    fake_features_num_samples,
):
    if real_features_num_samples < 2 or fake_features_num_samples < 2:
        raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
    mean_real = (real_features_sum / real_features_num_samples).unsqueeze(0)
    mean_fake = (fake_features_sum / fake_features_num_samples).unsqueeze(0)

    cov_real_num = real_features_cov_sum - real_features_num_samples * ops.mm(
        ops.transpose(mean_real, (1, 0)), mean_real
    )
    cov_real = cov_real_num / (real_features_num_samples - 1)
    cov_fake_num = fake_features_cov_sum - fake_features_num_samples * ops.mm(
        ops.transpose(mean_fake, (1, 0)), mean_fake
    )
    cov_fake = cov_fake_num / (fake_features_num_samples - 1)
    return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)


def _check_input_images(real_images, fake_images):
    return _check_input_images_helper(real_images), _check_input_images_helper(fake_images)


def _check_input_images_helper(images):
    if isinstance(images, Tensor):
        if images.dim() != 4:
            raise ValueError("The input tensor images must have 4 dimensions.")
        elif images.shape[0] < 2:
            raise ValueError("The input real images must have at least two samples.")
        else:
            return images
    elif isinstance(images, List):
        if len(images) < 2:
            raise ValueError("The input real images must have at least two samples.")
        elif not isinstance(images[0], Image.Image):
            raise TypeError("The input images list element must be an Image.")
        else:
            reformed_images = []
            for image in images:
                reformed_image = Tensor(np.array(image).transpose((2, 0, 1))).to(mindspore.uint8)
                reformed_images.append(reformed_image)
            return ops.stack(reformed_images, axis=0)
    else:
        raise TypeError("The input images must be a Tensor or a list of Images.")


def fid(
    real_images: Union[Tensor, List[Image.Image]],
    fake_images: Union[Tensor, List[Image.Image]],
    feature: int = 2048,
    normalize: bool = False,
    batch_size: int = -1,
):
    r"""Calculate Fréchet inception distance (FID_) which is used to access the quality of generated images.

    .. math::
        FID = \|\mu - \mu_w\|^2 + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})
    where :math:`\mathcal{N}(\mu, \Sigma)` is the multivariate normal distribution estimated from Inception v3
    (`fid ref1`_) features calculated on real life images and :math:`\mathcal{N}(\mu_w, \Sigma_w)` is the
    multivariate normal distribution estimated from Inception v3 features calculated on generated (fake) images.
    The metric was originally proposed in `fid ref1`_.

    Using the default feature extraction (Inception v3 using the original weights from `fid ref2`_), the input is
    expected to be mini-batches of 3-channel RGB images of shape ``(3xHxW)``. If argument ``normalize``
    is ``True`` images are expected to be dtype ``float`` and have values in the ``[0,1]`` range, else if
    ``normalize`` is set to ``False`` images are expected to have dtype ``uint8`` and take values in the ``[0, 255]``
    range. All images will be resized to 299 x 299 which is the size of the original training data. The boolian
    flag ``real`` determines if the images should update the statistics of the real distribution or the
    fake distribution.


    Args:
        real_images: Input real img tensors to evaluate.

        fake_images: Input fake img tensors to evaluate.

        feature:
            an integer:

            - an integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              64, 192, 768, 2048

        normalize:
            Argument for controlling the input image dtype normalization:

            - Controls whether input imgs have values in range [0, 1] or not:

              - True: if input imgs have values ranged in [0, 1]. They are cast to int8/byte tensors.
              - False: if input imgs have values ranged in [0, 255]. No casting is done.

        batch_size: batch size to separate the input data

    Returns:
        float scalar tensor with mean FID value over samples

    Raises:
        ValueError:
            If ``feature`` is set to an ``int`` not in [64, 192, 768, 2048]
        TypeError:
            If ``feature`` is not an ``str`` or ``int``
        ValueError:
            If ``reset_real_features`` is not an ``bool``

    Examples:
        >>> import mindspore as ms
        >>> from mindone.metrics.functional import fid
        >>> imgs_dist1 = ms.ops.randint(0, 200, (100, 3, 299, 299), seed=123).to(ms.uint8)
        >>> imgs_dist2 = ms.ops.randint(100, 255, (100, 3, 299, 299), seed=123).to(ms.uint8)
        >>> output = fid(imgs_dist1, imgs_dist2, feature=64)
        12.646194685028123
        hint: see README.md under metrics folder for more using cases

    """

    real_images, fake_images = _check_input_images(real_images, fake_images)
    inception = _get_feature_extractor(feature)
    if batch_size <= 0:
        logger.info("Batch size less than or equal to 0. Updating all images at once")
        real_features_sum, real_features_cov_sum, real_features_num_samples = _update_images(
            real_images, inception, normalize
        )
        fake_features_sum, fake_features_cov_sum, fake_features_num_samples = _update_images(
            fake_images, inception, normalize
        )

    else:
        logger.info(f"Updating images with batch size {batch_size}")
        real_features_sum, real_features_cov_sum, real_features_num_samples = _generate_origin_states(feature)
        fake_features_sum, fake_features_cov_sum, fake_features_num_samples = _generate_origin_states(feature)
        for chunk in ops.split(real_images, batch_size):
            raw_sum, cov_sum, num_samples = _update_images(chunk, inception, normalize)
            real_features_sum += raw_sum
            real_features_cov_sum += cov_sum
            real_features_num_samples += num_samples

        for chunk in ops.split(fake_images, batch_size):
            raw_sum, cov_sum, num_samples = _update_images(chunk, inception, normalize)
            fake_features_sum += raw_sum
            fake_features_cov_sum += cov_sum
            fake_features_num_samples += num_samples

    return _compute(
        real_features_sum,
        real_features_cov_sum,
        real_features_num_samples,
        fake_features_sum,
        fake_features_cov_sum,
        fake_features_num_samples,
    )
