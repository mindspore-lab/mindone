from typing import List, Union

from transformers import CLIPProcessor as _CLIPProcessor
from typing_extensions import Literal

import mindspore.ops as ops
from mindspore import Tensor

from mindone.metrics.functional.multimodal.clip_score import _get_clip_model_and_processor
from mindone.transformers import CLIPModel as _CLIPModel


def _encode_image(preprocessed_image, clip_model):
    image_features = clip_model.get_image_features(Tensor.from_numpy(preprocessed_image.numpy()))
    image_features = image_features / ops.norm(image_features, ord=2, dim=-1, keepdim=True)
    return image_features


def _encode_text(tokenized_text, clip_model):
    text_features = clip_model.get_text_features(Tensor.from_numpy(tokenized_text.numpy()))
    text_features = text_features / ops.norm(text_features, ord=2, dim=-1, keepdim=True)
    return text_features


def _clip_directional_similarity_update(
    image1: Tensor,
    image2: Tensor,
    text1: Union[str, List[str]],
    text2: Union[str, List[str]],
    model: _CLIPModel,
    processor: _CLIPProcessor,
):
    input_valid = False
    if isinstance(text1, str) and isinstance(text2, str) and image1.ndim == 3 and image2.ndim == 3:
        input_valid = True

    if isinstance(text1, list) and isinstance(text2, list) and image1.ndim == 4 and image2.ndim == 4:
        if len(text1) == len(text2) == image1.shape[0] == image2.shape[0]:
            input_valid = True

    if not input_valid:
        raise ValueError("Expected the number of images and text examples to be the same")

    processed_input1 = processor(text1, image1.asnumpy(), return_tensors="pt")
    processed_input2 = processor(text2, image2.asnumpy(), return_tensors="pt")
    image1_feature = _encode_image(processed_input1["pixel_values"], model)
    text1_feature = _encode_text(processed_input1["input_ids"], model)
    image2_feature = _encode_image(processed_input2["pixel_values"], model)
    text2_feature = _encode_text(processed_input2["input_ids"], model)
    return image1_feature, image2_feature, text1_feature, text2_feature


def clip_directional_similarity(
    origin_image: Tensor,
    generated_image: Tensor,
    origin_text: Union[str, List[str]],
    edited_text: Union[str, List[str]],
    model_name_or_path: Literal[
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14-336",
        "openai/clip-vit-large-patch14",
    ] = "openai/clip-vit-large-patch14",
) -> Tensor:
    r"""Calculates `CLIP Directional Similarity`_ to measure the consistency of the change between the two images
    (in CLIP space) with the change between the two image captions.
    The metric was originally proposed in `clip_dir_sim ref1`_.

    .. math::
        \text{CLIPDirectionalSimilarity(I1, I2, C1, C2)} = max(100 * cos(E_I1 - E_I2, E_C1 - E_C2), 0)

    which corresponds to the cosine similarity between the difference of visual `CLIP`_ embeddings of two images and
    textual CLIP embeddings of two texts. The higher the CLIP directional similarity, the better it is.

    .. note:: Clip Directional Similarity metric does not support GRAPH_MODE

    Args:
        origin_image: Tensor, origin image tensor
        generated_image: Tensor, generated image tensor
        origin_text: Union[str, List[str]], origin texts
        edited_text: Union[str, List[str]], edited texts
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are:

            - `"openai/clip-vit-base-patch16"`
            - `"openai/clip-vit-base-patch32"`
            - `"openai/clip-vit-large-patch14-336"`
            - `"openai/clip-vit-large-patch14"`

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindone.metrics.functional import clip_directional_similarity
        >>> np.random.seed(123)
        >>> origin_image_np = np.random.randint(0, 255, (3, 224, 224))
        >>> generated_image_np = np.random.randint(0, 255, (3, 224, 224))
        >>> origin_text = "a photo of cat"
        >>> edited_text = "a photo of dog"
        >>> original_image_ms = ms.Tensor(origin_image_np).to(ms.uint8)
        >>> original_caption_ms = origin_text
        >>> edited_image_ms = ms.Tensor(generated_image_np).to(ms.uint8)
        >>> modified_caption_ms = edited_text
        >>> clip_directional_similarity(original_image_ms, edited_image_ms, original_caption_ms, modified_caption_ms)
        -0.028463803
        note: the output may be different since features extracted from clip model are different. We're trying to
        fix this problem with mindnlp developers.

    """
    model, processor = _get_clip_model_and_processor(model_name_or_path)
    img_feat_one, img_feat_two, text_feat_one, text_feat_two = _clip_directional_similarity_update(
        origin_image, generated_image, origin_text, edited_text, model, processor
    )

    sim_direction = ops.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
    return ops.mean(sim_direction)
