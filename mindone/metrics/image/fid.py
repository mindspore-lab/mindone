from typing import Any

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from mindone.metrics.metric import Metric
from mindone.metrics.utils.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3


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


class FrechetInceptionDistance(Metric):
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

    This metric is known to be unstable in its calculations, as generating different answers (relative difference larger
    than 1e-3), so calculation using mindspore.float64 is recommended. Unfortunately, ops.AllReduce does not support
    mindspore.float64. So distributed calculation may generate a different result.

    As input to ``construct`` and ``update`` the metric accepts the following input

    - ``imgs`` (:class:`~mindspore.Tensor`): tensor with images feed to the feature extractor with
    - ``real`` (:class:`~bool`): bool indicating if ``imgs`` belong to the real or the fake distribution

    As output of `construct` and `compute` the metric returns the following output

    - ``fid`` (:class:`~mindspore.Tensor`): float scalar tensor with mean FID value over samples

    Args:
        feature:
            an integer:

            - an integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              64, 192, 768, 2048

        reset_real_features: Whether to also reset the real features. Since in many cases the real dataset does not
            change, the features can be cached them to avoid recomputing them which is costly. Set this to ``False`` if
            your dataset does not change.
        normalize:
            Argument for controlling the input image dtype normalization:

            - Controls whether input imgs have values in range [0, 1] or not:

              - True: if input imgs have values ranged in [0, 1]. They are cast to int8/byte tensors.
              - False: if input imgs have values ranged in [0, 255]. No casting is done.

        kwargs: Additional keyword arguments, passed to parent class mindspore.nn.Cell directly.

    Raises:
        ValueError:
            If ``feature`` is set to an ``int`` not in [64, 192, 768, 2048]
        TypeError:
            If ``feature`` is not an ``str`` or ``int``
        ValueError:
            If ``reset_real_features`` is not an ``bool``

    Examples:
        >>> import mindspore as ms
        >>> from mindone.metrics.fid import FrechetInceptionDistance
        >>> fid = FrechetInceptionDistance(feature=64)
        >>> imgs_dist1 = ms.ops.randint(0, 200, (100, 3, 299, 299), seed=123).to(ms.uint8)
        >>> imgs_dist2 = ms.ops.randint(100, 255, (100, 3, 299, 299), seed=123).to(ms.uint8)
        >>> fid.update(imgs_dist1, real=True)
        >>> fid.update(imgs_dist2, real=False)
        >>> fid.compute()
        12.646194685028123
        hint: see README.md under metrics folder for more using cases

    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    inception: nn.Cell
    feature_network: str = "inception"

    def __init__(
        self, feature: int = 2048, reset_real_features: bool = True, normalize: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(normalize, bool):
            raise TypeError("Argument `normalize` expected to be a bool")
        self.normalize = normalize
        self.used_custom_model = False

        if isinstance(feature, int):
            num_features = feature
            valid_int_input = (64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )

            self.inception = FeatureExtractorInceptionV3(
                name="inecption-v3-compat",
                request_feature=str(feature),
            )
        else:
            raise TypeError("Got unknown input to argument `feature`")

        self.feature = feature

        if not isinstance(reset_real_features, bool):
            raise TypeError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        mx_num_feats = (num_features, num_features)

        self.real_features_sum = ops.zeros(num_features).to(mindspore.float32)
        self.real_features_cov_sum = ops.zeros(mx_num_feats).to(mindspore.float32)
        self.real_features_num_samples = mindspore.tensor(0).long()
        self.fake_features_sum = ops.zeros(num_features).to(mindspore.float32)
        self.fake_features_cov_sum = ops.zeros(mx_num_feats).to(mindspore.float32)
        self.fake_features_num_samples = mindspore.tensor(0).long()

        self.all_reduce = ops.AllReduce(ops.ReduceOp.SUM)

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features.

        Args:
            imgs: Input img tensors to evaluate.
            real: Whether given image is real or fake.
        """
        imgs = (imgs * 255).to(mindspore.uint8) if self.normalize else imgs
        features = self.inception(imgs)
        features = features.to(mindspore.double)
        if real:
            self.real_features_sum += ops.sum(features, dim=0)
            self.real_features_cov_sum += ops.mm(ops.transpose(features, (1, 0)), features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += ops.sum(features, dim=0)
            self.fake_features_cov_sum += ops.mm(ops.transpose(features, (1, 0)), features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * ops.mm(
            ops.transpose(mean_real, (1, 0)), mean_real
        )
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * ops.mm(
            ops.transpose(mean_fake, (1, 0)), mean_fake
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)

    def _generate_features(self, imgs: Tensor):
        imgs = (imgs * 255).to(mindspore.uint8) if self.normalize else imgs
        features = self.inception(imgs)
        # ops.AllReduce does not support mindspore.float64
        if mindspore.communication.GlobalComm.INITED:
            features = features.to(mindspore.float32)
        else:
            features = features.to(mindspore.double)

        if features.dim() == 1:
            features = features.unsqueeze(0)
        return features

    def reset(self) -> None:
        """Reset metric states."""
        num_features = self.feature
        mx_num_feats = (num_features, num_features)
        self.fake_features_sum = ops.zeros(num_features).to(mindspore.float32)
        self.fake_features_cov_sum = ops.zeros(mx_num_feats).to(mindspore.float32)
        self.fake_features_num_samples = mindspore.tensor(0).long()
        if self.reset_real_features:
            self.real_features_sum = ops.zeros(num_features).to(mindspore.float32)
            self.real_features_cov_sum = ops.zeros(mx_num_feats).to(mindspore.float32)
            self.real_features_num_samples = mindspore.tensor(0).long()
