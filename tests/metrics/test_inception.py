import mindspore
import mindspore.nn as nn
import numpy as np
import pytest
import torch
from torchmetrics.image.inception import InceptionScore as TorchInceptionScore
from PIL import Image

from mindone.metrics.inception_score import InceptionScore
from mindone.metrics.functional.inception_score import inception_score as inception_score_func


def test_no_train():
    """Assert that metric never leaves evaluation mode."""

    class MyModel(nn.Cell):
        def __init__(self) -> None:
            super().__init__()
            self.metric = InceptionScore()

        def construct(self, x):
            return x

    model = MyModel()
    model.set_train()

    assert model.training
    assert (
        not model.metric.inception.training
    ), "InceptionScore metric was changed to training mode which should not happen"


def test_is_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    with pytest.raises(ValueError, match="Input to argument `feature` must be one of .*"):
        _ = InceptionScore(feature=2)

    with pytest.raises(TypeError, match="Got unknown input to argument `feature`"):
        # noinspection PyTypeChecker
        InceptionScore(feature=[1, 2])


# noinspection PyTypeChecker
def test_is_func_input():
    """Test different inputs."""
    image_np = np.random.randint(0, 255, (10, 3, 299, 299))
    images_tensor = mindspore.Tensor(image_np).to(mindspore.uint8)
    images_list = []
    for idx in range(10):
        real_image = Image.fromarray(image_np[idx].transpose((1, 2, 0)).astype(np.uint8))
        images_list.append(real_image)

    is_tensor_input, _ = inception_score_func(images_tensor)
    is_list_input, _ = inception_score_func(images_list)
    assert mindspore.ops.isclose(is_tensor_input, is_list_input, rtol=1e-3)

    fid_with_batch, _ = inception_score_func(images_tensor, batch_size=3)
    assert mindspore.ops.isclose(is_tensor_input, fid_with_batch, rtol=1e-3)

    with pytest.raises(TypeError):
        inception_score_func(1)

    with pytest.raises(ValueError):
        inception_score_func(mindspore.ops.randint(0, 255, (1, 3, 255, 255)))

    with pytest.raises(ValueError):
        inception_score_func(mindspore.ops.randint(0, 255, (3, 255, 255)))

    with pytest.raises(ValueError):
        inception_score_func([Image.fromarray(image_np[0].transpose((1, 2, 0)).astype(np.uint8))])

    with pytest.raises(TypeError):
        inception_score_func([mindspore.ops.randint(0, 255, (3, 255, 255)) for _ in range(10)])


def test_is_update_compute():
    """Test that inception score works as expected."""
    metric = InceptionScore()

    for _ in range(2):
        np_array = np.random.randint(0, 255, (100, 3, 299, 299))
        img = mindspore.Tensor(np_array).to(mindspore.uint8)
        metric.update(img)

    mean, std = metric.compute()
    assert mean >= 0.0
    assert std >= 0.0


def test_functional():
    np_array = np.random.randint(0, 255, (100, 3, 299, 299))
    img = mindspore.Tensor(np_array).to(mindspore.uint8)
    mean, std = inception_score_func(img)
    assert mean >= 0.0
    assert std >= 0.0


@pytest.mark.parametrize("normalize", [True, False])
def test_compare_is(normalize):
    """Check that the hole pipeline give the same result as torchmetrics."""

    np_array = np.random.randint(0, 255, (100, 3, 299, 299))

    # torchmetrics calculation
    tensor_inception = torch.from_numpy(np_array).to(torch.uint8)
    inception_torch = TorchInceptionScore(splits=1)
    inception_torch.update(tensor_inception)
    torch_mean, _ = inception_torch.compute()

    # mindone calculation
    tensor_inception_ms = mindspore.Tensor(np_array).to(mindspore.uint8)
    inception = InceptionScore(splits=1)
    inception.update(tensor_inception_ms)
    mindone_mean, _ = inception.compute()

    # mindone functional calculation
    mindone_func_mean, _ = inception_score_func(tensor_inception_ms, splits=1)

    assert np.allclose(torch_mean.numpy(), mindone_mean.asnumpy(), rtol=1e-3)
    assert np.allclose(torch_mean.numpy(), mindone_func_mean.asnumpy(), rtol=1e-3)
