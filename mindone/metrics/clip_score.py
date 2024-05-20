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

from typing import Any, Union, List

import mindspore as ms
from mindspore import ops, Tensor
from mindspore.nn import Cell
from typing_extensions import Literal

from mindone.metrics.functional.clip_score import _get_clip_model_and_processor, _clip_score_update


class ClipScore(Cell):
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

            - `"clip_vit_b_16"`
            - `"clip_vit_b_32"`
            - `"clip_vit_l_14@336"`
            - `"clip_vit_l_14"`

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
                "clip_vit_b_16",
                "clip_vit_b_32",
                "clip_vit_l_14@336",
                "clip_vit_l_14",
            ] = "clip_vit_l_14",
            **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model, self.processor = _get_clip_model_and_processor(pretrained_model)
        self.score = ops.zeros(1).to(ms.float32)
        self.n_samples = ops.zeros(1).to(ms.int32)
        self.all_reduce = ops.AllReduce(op=ops.ReduceOp.SUM)
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

    def construct(self, images, text):
        score = ops.zeros(1).to(ms.float32)
        n_samples = ops.zeros(1).to(ms.int32)
        processed_input = self.processor(text_input=text, image_input=images)

        img_features = self.model.get_image_features(processed_input['image'])
        img_features = img_features / ops.norm(img_features, ord=2, dim=-1, keepdim=True)

        txt_features = self.model.get_text_features(processed_input['text'])
        txt_features = txt_features / ops.norm(txt_features, ord=2, dim=-1, keepdim=True)

        # cosine similarity between feature vectors
        score += ops.sum(ops.sum(100 * (img_features * txt_features), dim=-1))

        n_samples += len(text) if isinstance(text, list) else 1

        if ms.communication.GlobalComm.INITED:
            score = self.all_reduce(score)
            n_samples = self.all_reduce(n_samples)

        out = score / n_samples
        if not ops.less(0, out):
            return ops.zeros_like(out)

        return out

    def reset(self):
        self.score = ops.zeros(1).to(ms.float32)
        self.n_samples = ops.zeros(1).to(ms.int32)
