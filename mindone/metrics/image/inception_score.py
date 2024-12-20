import logging
from typing import Any, List, Tuple, Union

import mindspore
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.nn import Cell

from mindone.metrics.metric import Metric
from mindone.metrics.utils.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

logger = logging.getLogger(__name__)


class InceptionScore(Metric):
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

    As input to ``construct`` and ``update`` the metric accepts the following input

    - ``imgs`` (:class:`~mindspore.Tensor`): tensor with images feed to the feature extractor

    As output of `forward` and `compute` the metric returns the following output

    - ``inception_mean`` (:class:`~mindspore.Tensor`): float scalar tensor with mean inception score over subsets
    - ``inception_std`` (:class:`~mindspore.Tensor`): float scalar tensor with standard deviation of inception score
      over subsets

    Args:
        features:
            Either a str or an integer (custom feature extractor is currently not supported, please raise issue if you
             prefer this feature to be done):

            - an str or integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              'logits_unbiased', 64, 192, 768, 2048

        splits: integer determining how many splits the inception score calculation should be split among
        kwargs: Additional keyword arguments, passed to parent class mindspore.nn.Cell directly.

    Raises:
        ValueError:
            If ``feature`` is set to an ``str`` or ``int`` and not one of ``('logits_unbiased', 64, 192, 768, 2048)``
        TypeError:
            If ``feature`` is not an ``str``, ``int``

    Examples:
        >>> import mindspore as ms
        >>> from mindone.metrics.inception_score import InceptionScore
        >>> # splits equals to 1 to avoid shuffle operation generating different results
        >>> inception_score = InceptionScore(splits=1)
        >>> imgs = ms.ops.randint(0, 255, (100, 3, 299, 299), seed=123).to(ms.uint8)
        >>> inception_score.update(imgs)
        >>> inception_score.compute()
        (Tensor(shape=[], dtype=Float32, value= 1.06732), Tensor(shape=[], dtype=Float32, value= 0))
        hint: see README.md under metrics folder for more using cases

    """
    features: List
    inception: Cell

    def __init__(
        self,
        feature: Union[str, int] = "logits_unbiased",
        splits: int = 10,
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        logger.warning(
            "Metric `InceptionScore` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint."
        )

        valid_inputs = ("logits_unbiased", 64, 192, 768, 2048)
        data_type = self._dtype if hasattr(self, "_dtype") else mindspore.float32
        if isinstance(feature, (str, int)):
            if feature not in valid_inputs:
                raise ValueError(f"Input to argument `feature` must be one of {valid_inputs}, but got {feature}.")
            self.inception = FeatureExtractorInceptionV3(
                name="inecption-v3-compat", request_feature=str(feature), custom_dtype=data_type
            )
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        self.splits = splits

        if mindspore.communication.GlobalComm.INITED:
            self.all_gather = mindspore.ops.AllGather()
            print(f"Initializing Ascend context, rank id: {mindspore.communication.get_rank()}")
        self.features = []
        self.print = ops.Print()

    def update(self, imgs: Tensor) -> None:
        """Update the state with extracted features.

        Args:
            imgs: Input img tensors to evaluate

        """
        imgs = (imgs * 255).to(mindspore.uint8) if self.normalize else imgs
        out = self.inception(imgs)
        features = out.reshape(imgs.shape[0], -1)
        self.features.append(features)

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if isinstance(self.features, mindspore.Tensor):
            features = self.features
        else:
            features = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in self.features]
            if not features:  # empty list
                raise ValueError("No samples to concatenate")
            features = ops.cat(features, axis=0)

        # random shuffle the features
        features = ops.shuffle(features)

        # calculate probs and logits
        prob = ops.softmax(features, axis=1)
        log_prob = ops.log_softmax(features, axis=1)

        # split into groups

        prob = ops.chunk(prob, chunks=self.splits, axis=0)
        log_prob = ops.chunk(log_prob, chunks=self.splits, axis=0)

        # calculate score per split
        mean_prob = [ops.mean(p, axis=0, keep_dims=True) for p in prob]
        kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [ops.sum(k, dim=1).mean().exp() for k in kl_]
        kl = ops.stack(kl_)

        # return mean and std
        return kl.mean(), kl.std()
