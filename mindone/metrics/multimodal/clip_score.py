from typing import Any, List, Union

from typing_extensions import Literal

import mindspore as ms
from mindspore import Tensor, ops

from mindone.metrics.functional.multimodal.clip_score import _clip_score_update, _get_clip_model_and_processor
from mindone.metrics.metric import Metric


class ClipScore(Metric):
    r"""Calculates `CLIP Score`_ which is a text-to-image similarity metric.

    CLIP Score is a reference free metric that can be used to evaluate the correlation between a generated caption for
    an image and the actual content of the image. It has been found to be highly correlated with human judgement. The
    metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual `CLIP`_ embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the closer
    to 100 the better.

    .. note:: Clip Score metric does not support GRAPH_MODE

    As input to ``construct`` and ``update`` the metric accepts the following input

    - ``images`` (:class:`~mindspore.Tensor` or list of tensors): tensor with images feed to the feature extractor with.
        If a single tensor it should have shape ``(N, C, H, W)``. If a list of tensors, each tensor should have shape
        ``(C, H, W)``. ``C`` is the number of channels, ``H`` and ``W`` are the height and width of the image.
    - ``text`` (:class:`~str` or :class:`~list` of :class:`~str`): text to compare with the images, one for each image.

    As output of `construct` and `compute` the metric returns the following output

    - ``clip_score`` (:class:`~mindspore.Tensor`): float scalar tensor with mean CLIP score over samples

    Args:
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are:

            - `"openai/clip-vit-base-patch16"`
            - `"openai/clip-vit-base-patch32"`
            - `"openai/clip-vit-large-patch14-336"`
            - `"openai/clip-vit-large-patch14"`

        kwargs: Additional keyword arguments, passed to parent class mindspore.nn.Cell directly.

    Examples:
        >>> import mindspore as ms
        >>> from mindone.metrics.clip_score import ClipScore
        >>> images = ms.ops.randint(0, 255, (3, 244, 244), seed=123).to(ms.uint8)
        >>> text = "a photo of a cat"
        >>> metric = ClipScore()
        >>> metric.update(images, text)
        >>> metric.compute()
        [20.188786]
        note: the output may be different since features extracted from clip model are different. We're trying to fix
        this problem with mindnlp developers.

    """

    def __init__(
        self,
        pretrained_model: Literal[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14-336",
            "openai/clip-vit-large-patch14",
        ] = "openai/clip-vit-large-patch14",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model, self.processor = _get_clip_model_and_processor(pretrained_model)
        self.score = ops.zeros(1).to(ms.float32)
        self.n_samples = ops.zeros(1).to(ms.int32)
        self.to_float(ms.float32)

    def update(self, images: Tensor, text: Union[str, List[str]]) -> None:
        score, n_samples = _clip_score_update(images, text, self.model, self.processor)
        self.score += ops.sum(score)
        self.n_samples += n_samples

    def compute(self) -> Tensor:
        output = self.score / self.n_samples
        if not ops.less(0, output):
            return ops.zeros_like(output)

        return output

    def reset(self):
        self.score = ops.zeros(1).to(ms.float32)
        self.n_samples = ops.zeros(1).to(ms.int32)
