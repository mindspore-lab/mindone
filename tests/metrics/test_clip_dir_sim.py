import pytest

import mindspore as ms

from mindone.metrics.functional.multimodal.clip_directional_similarity import clip_directional_similarity
from mindone.metrics.multimodal.clip_directional_similarity import ClipDirectionalSimilarity


def test_clip_dir_sim():
    imgs1 = ms.ops.randint(0, 255, (3, 3, 244, 244)).to(ms.uint8)
    imgs2 = ms.ops.randint(0, 255, (3, 3, 244, 244)).to(ms.uint8)
    texts1 = ["an image of dog" for _ in range(3)]
    texts2 = ["an image of cat" for _ in range(3)]
    metric = ClipDirectionalSimilarity()
    metric.update(imgs1, imgs2, texts1, texts2)
    output = metric.compute()
    assert output.shape == ()

    img1 = ms.ops.randint(0, 255, (3, 244, 244)).to(ms.uint8)
    img2 = ms.ops.randint(0, 255, (3, 244, 244)).to(ms.uint8)
    text1 = "an image of dog"
    text2 = "an image of cat"
    metric = ClipDirectionalSimilarity()
    metric.update(img1, img2, text1, text2)
    output = metric.compute()
    assert output.shape == ()


def test_clip_dir_sim_func():
    imgs1 = ms.ops.randint(0, 255, (3, 3, 244, 244)).to(ms.uint8)
    imgs2 = ms.ops.randint(0, 255, (3, 3, 244, 244)).to(ms.uint8)
    texts1 = ["an image of dog" for _ in range(3)]
    texts2 = ["an image of cat" for _ in range(3)]
    output = clip_directional_similarity(imgs1, imgs2, texts1, texts2)
    assert output.shape == ()


def test_error_on_invalid_input():
    imgs1 = ms.ops.randint(0, 255, (3, 244, 244)).to(ms.uint8)
    imgs2 = ms.ops.randint(0, 255, (5, 3, 244, 244)).to(ms.uint8)
    texts1 = ["an image of dog" for _ in range(5)]
    texts2 = ["an image of cat" for _ in range(5)]
    with pytest.raises(ValueError):
        metric = ClipDirectionalSimilarity()
        metric.reset()
        metric.update(imgs1, imgs2, texts1, texts2)
        metric.compute()

    imgs1 = ms.ops.randint(0, 255, (5, 3, 244, 244)).to(ms.uint8)
    imgs2 = ms.ops.randint(0, 255, (5, 3, 244, 244)).to(ms.uint8)
    texts1 = "an image of dog"
    texts2 = ["an image of cat" for _ in range(5)]
    with pytest.raises(ValueError):
        metric = ClipDirectionalSimilarity()
        metric.reset()
        metric.update(imgs1, imgs2, texts1, texts2)
        metric.compute()

    imgs1 = ms.ops.randint(0, 255, (5, 3, 244, 244)).to(ms.uint8)
    imgs2 = ms.ops.randint(0, 255, (5, 3, 244, 244)).to(ms.uint8)
    texts1 = ["an image of cat" for _ in range(3)]
    texts2 = ["an image of cat" for _ in range(5)]
    with pytest.raises(ValueError):
        metric = ClipDirectionalSimilarity()
        metric.reset()
        metric.update(imgs1, imgs2, texts1, texts2)
        metric.compute()
