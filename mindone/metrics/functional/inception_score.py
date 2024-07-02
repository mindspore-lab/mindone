# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import logging
from typing import Union, List

import mindspore
import mindspore.ops as ops
from PIL import Image
from mindspore import Tensor

from mindone.metrics.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3 as _FeatureExtractorInceptionV3
from mindone.metrics.functional.fid import _check_input_images_helper

logger = logging.getLogger("mindone.metrics.functional.inception_score")


def _get_feature_extractor(feature: str):
    inception = _FeatureExtractorInceptionV3(
        name="inecption-v3-compat",
        request_feature=feature
    )
    return inception


def _check_input_images(images):
    return _check_input_images_helper(images)


def _update_images(images, inception, normalize):
    images = (images * 255).to(mindspore.uint8) if normalize else images
    out = inception(images)
    features = out.reshape(images.shape[0], -1)
    return features


def _compute(features, splits):
    if isinstance(features, Tensor):
        features = features
    else:
        features = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in features]
        if not features:  # empty list
            raise ValueError("No samples to concatenate")
        features = ops.cat(features, axis=0)

    # random permute the features
    features = ops.shuffle(features)

    # calculate probs and logits
    prob = ops.softmax(features, axis=1)
    log_prob = ops.log_softmax(features, axis=1)

    # split into groups

    prob = ops.chunk(prob, chunks=splits, axis=0)
    log_prob = ops.chunk(log_prob, chunks=splits, axis=0)

    # calculate score per split
    mean_prob = [ops.mean(p, axis=0, keep_dims=True) for p in prob]
    kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
    kl_ = [ops.sum(k, dim=1).mean().exp() for k in kl_]
    kl = ops.stack(kl_)

    # return mean and std
    return kl.mean(), kl.std()


def inception_score(
    images: Union[Tensor, List[Image.Image]],
    feature: str = 'logits_unbiased',
    batch_size: int = -1,
    normalize: bool = False,
    splits: int = 10
):
    r"""Calculate the Inception Score (IS) which is used to access how realistic generated images are.

    .. math::
        IS = exp(\mathbb{E}_x KL(p(y | x ) || p(y)))

    where :math:`KL(p(y | x) || p(y))` is the KL divergence between the conditional distribution :math:`p(y|x)`
    and the marginal distribution :math:`p(y)`. Both the conditional and marginal distribution is calculated
    from features extracted from the images. The score is calculated on random splits of the images such that
    both a mean and standard deviation of the score are returned. The metric was originally proposed in
    `inception ref1`_.

    Using the default feature extraction (Inception v3 using the original weights from `inception ref2`_), the input
    is expected to be mini-batches of 3-channel RGB images of shape ``(3xHxW)``. If argument ``normalize``
    is ``True`` images are expected to be dtype ``float`` and have values in the ``[0,1]`` range, else if
    ``normalize`` is set to ``False`` images are expected to have dtype uint8 and take values in the ``[0, 255]``
    range. All images will be resized to 299 x 299 which is the size of the original training data.

    .. note:: using this metric with the default feature extractor requires connection with URL:
        https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth
        weights will be automatically transferred with script 'convert_inception_weights.py'

    Args:
        images: Input img tensors to evaluate.

        feature:
            an integer:

            - an integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              64, 192, 768, 2048

        batch_size: batch size to separate the input data

        normalize:
            Argument for controlling the input image dtype normalization:

            - Controls whether input imgs have values in range [0, 1] or not:

              - True: if input imgs have values ranged in [0, 1]. They are cast to int8/byte tensors.
              - False: if input imgs have values ranged in [0, 255]. No casting is done.

        splits: integer determining how many splits the inception score calculation should be split among

    Returns:
        - ``inception_mean`` (:class:`~mindspore.Tensor`): float scalar tensor with mean inception score over subsets
        - ``inception_std`` (:class:`~mindspore.Tensor`): float scalar tensor with standard deviation of inception score
          over subsets

    Raises:
        ValueError:
            If ``feature`` is set to an ``str`` or ``int`` and not one of ``('logits_unbiased', 64, 192, 768, 2048)``
        TypeError:
            If ``feature`` is not an ``str``, ``int``

    Examples:
        >>> import mindspore as ms
        >>> from mindone.metrics.functional import inception_score
        >>>  # splits equals to 1 to avoid shuffle operation generating different results
        >>> imgs = ms.ops.randint(0, 255, (100, 3, 299, 299), seed=123).to(ms.uint8)
        >>> inception_score(imgs, splits=1)
        (Tensor(shape=[], dtype=Float32, value= 1.06732), Tensor(shape=[], dtype=Float32, value= 0))
        hint: see README.md under metrics folder for more using cases

    """
    images = _check_input_images(images)
    inception = _get_feature_extractor(feature)
    if batch_size <= 0:
        logger.info(f"Batch size less than or equal to 0. Updating all images at once")
        features = _update_images(images, inception, normalize)
    else:
        logger.info(f"Updating images with batch size {batch_size}")
        features = []
        for chunk in ops.split(images, batch_size):
            batch_features = _update_images(chunk, inception, normalize)
            features.append(batch_features)
    return _compute(features, splits)

        