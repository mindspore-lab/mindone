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

from mindformers import CLIPModel as _CLIPModel
from mindformers import CLIPProcessor as _CLIPProcessor
from typing_extensions import Literal
from typing import List, Tuple, Union
from mindspore import Tensor
import logging
import mindspore.ops as ops
import mindspore as ms

logger = logging.getLogger("mindone.metrics.functional.inception_score")


def _clip_score_update(
    images: Tensor,
    text: Union[str, List[str]],
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> Tuple[Tensor, int]:
    if isinstance(text, list) and images.ndim == 4:
        if len(text) != images.shape[0]:
            raise ValueError(
                f"Expected the number of images and text examples to be the same but got {len(images)} and {len(text)}"
            )

    processed_input = processor(text_input=text, image_input=images)

    img_features = model.get_image_features(processed_input['image']).to(ms.float32)
    img_features = img_features / ops.norm(img_features, ord=2, dim=-1, keepdim=True)

    txt_features = model.get_text_features(processed_input['text']).to(ms.float32)
    txt_features = txt_features / ops.norm(txt_features, ord=2, dim=-1, keepdim=True)

    # cosine similarity between feature vectors
    score = 100 * (img_features * txt_features).sum(axis=-1)
    return score, len(text) if isinstance(text, list) else 1


def _get_clip_model_and_processor(
    model_name_or_path: Literal[
        "clip_vit_b_16",
        "clip_vit_b_32",
        "clip_vit_l_14@336",
        "clip_vit_l_14",
    ] = "clip_vit_l_14",
) -> Tuple[_CLIPModel, _CLIPProcessor]:

    model = _CLIPModel.from_pretrained(model_name_or_path).to_float(ms.float32)
    processor = _CLIPProcessor.from_pretrained(model_name_or_path)
    return model, processor


def clip_score(
    images: Tensor,
    text: Union[str, List[str]],
    model_name_or_path: Literal[
        "clip_vit_b_16",
        "clip_vit_b_32",
        "clip_vit_l_14@336",
        "clip_vit_l_14",
    ] = "clip_vit_l_14",
) -> Tensor:
    r"""Calculate `CLIP Score`_ which is a text-to-image similarity metric.

    CLIP Score is a reference free metric that can be used to evaluate the correlation between a generated caption
    for an image and the actual content of the image. It has been found to be highly correlated with human
    judgement. The metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual `CLIP`_ embedding :math:`E_i` for an image :math:`i`
    and textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the
    closer to 100 the better.

    .. note:: Metric is not scriptable

    Args:
        images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors
        text: Either a single caption or a list of captions
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are
        - `"clip_vit_b_16"`
        - `"clip_vit_b_32"`
        - `"clip_vit_l_14@336"`
        - `"clip_vit_l_14"`

    Raises:
        ValueError:
            If the number of images and captions do not match

    Example:
        >>> import mindspore as ms
        >>> from mindone.metrics.functional import clip_score
        >>> images = ms.ops.randint(0, 255, (3, 244, 244), seed=123).to(ms.uint8)
        >>> text = "a photo of a cat"
        >>> output = clip_score(images, text)
        [20.188786]
        note: the output may be different since features extracted from clip model are different. We're trying to fix
        this problem with mindnlp developers.

    """
    model, processor = _get_clip_model_and_processor(model_name_or_path)
    score, n_samples = _clip_score_update(images, text, model, processor)
    output = (ops.zeros(1).to(ms.float32) + ops.sum(score)) / n_samples
    if not ops.less(0, output):
        return ops.zeros_like(output)

    return output
