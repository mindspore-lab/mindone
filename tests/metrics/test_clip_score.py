
import pytest
import torch
import numpy as np
import mindspore as ms
from mindone.metrics.clip_score import ClipScore
from mindone.metrics.functional.clip_score import clip_score
from torchmetrics.multimodal.clip_score import CLIPScore as TorchCLIPScore

def test_clip_score():
    img = ms.ops.randint(0, 255, (3, 224, 224)).to(ms.uint8)
    texts = "an image of dog"
    metric = ClipScore()
    metric.update(img, texts)
    output = metric.compute()
    assert output.shape == (1,)

    imgs = ms.ops.randint(0, 255, (2, 3, 224, 224)).to(ms.uint8)
    texts = ["an image of dog" + str(idx) for idx in range(2)]
    metric.reset()
    metric.update(imgs, texts)
    output = metric.compute()
    assert output.shape == (1,)


def test_clip_score_func():
    imgs = ms.ops.randint(0, 255, (2, 3, 224, 224)).to(ms.uint8)
    texts = ["an image of dog" + str(idx) for idx in range(2)]
    output = clip_score(imgs, texts)
    assert output.shape == (1,)


# noinspection PyTypeChecker
def test_compare_clip_score():
    img_np = np.random.randint(0, 255, (3, 224, 224))
    img_torch = torch.from_numpy(img_np).to(torch.uint8)
    text = "an image of dog"
    torch_clip_score = TorchCLIPScore("openai/clip-vit-large-patch14")
    torch_clip_score.update(img_torch, text)
    torch_score = torch_clip_score.compute()

    img_ms = ms.Tensor(img_np).to(ms.uint8)
    ms_clip_score = ClipScore("clip_vit_l_14")
    ms_clip_score.update(img_ms, text)
    ms_score = ms_clip_score.compute()

    assert np.allclose(torch_score.numpy(), ms_score.asnumpy(), rtol=1e-3)

    ms_func_score = clip_score(img_ms, text, model_name_or_path="clip_vit_l_14")
    assert np.allclose(torch_score.numpy(), ms_func_score.asnumpy(), rtol=1e-3)


def test_error_on_not_same_amount_of_input():
    imgs = ms.ops.randint(0, 255, (3, 3, 224, 224)).to(ms.uint8)
    texts = ["an image of dog" + str(idx) for idx in range(2)]
    with pytest.raises(ValueError):
        metric = ClipScore()
        metric.reset()
        metric.update(imgs, texts)
        metric.compute()
