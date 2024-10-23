import numpy as np
import pytest
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance as TorchFrechetInceptionDistance

import mindspore
import mindspore.nn as nn

from mindone.metrics.functional.image.fid import fid as fid_func
from mindone.metrics.image.fid import FrechetInceptionDistance


def test_no_train():
    """Assert that metric never leaves evaluation mode."""

    class MyModel(nn.Cell):
        def __init__(self) -> None:
            super().__init__()
            self.metric = FrechetInceptionDistance()

        def construct(self, x):
            return x

    model = MyModel()
    model.set_train()

    assert model.training
    assert (
        not model.metric.inception.training
    ), "FrechetInceptionDistance metric was changed to training mode which should not happen"


# noinspection PyTypeChecker
def test_fid_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""

    with pytest.raises(TypeError, match="Argument `normalize` expected to be a bool"):
        FrechetInceptionDistance(normalize=1)

    with pytest.raises(ValueError, match="Integer input to argument `feature` must be one of .*"):
        FrechetInceptionDistance(feature=2)

    with pytest.raises(TypeError, match="Got unknown input to argument `feature`"):
        FrechetInceptionDistance(feature=[1, 2])

    with pytest.raises(TypeError, match="Argument `reset_real_features` expected to be a bool"):
        FrechetInceptionDistance(reset_real_features=1)


# noinspection PyTypeChecker
def test_fid_func_input():
    """Test different inputs."""
    real_image_np = np.random.randint(0, 200, (10, 3, 299, 299))
    fake_image_np = np.random.randint(0, 200, (10, 3, 299, 299))
    real_images_tensor = mindspore.Tensor(real_image_np).to(mindspore.uint8)
    fake_images_tensor = mindspore.Tensor(fake_image_np).to(mindspore.uint8)
    real_images_list = []
    fake_images_list = []
    for idx in range(10):
        real_image = Image.fromarray(real_image_np[idx].transpose((1, 2, 0)).astype(np.uint8))
        fake_image = Image.fromarray(fake_image_np[idx].transpose((1, 2, 0)).astype(np.uint8))
        real_images_list.append(real_image)
        fake_images_list.append(fake_image)

    fid_tensor_input = fid_func(real_images_tensor, fake_images_tensor)
    fid_list_input = fid_func(real_images_list, fake_images_list)
    assert mindspore.ops.isclose(fid_tensor_input, fid_list_input, rtol=1e-5)

    fid_with_batch = fid_func(real_images_tensor, fake_images_tensor, batch_size=3)
    assert mindspore.ops.isclose(fid_tensor_input, fid_with_batch, rtol=1e-5)

    with pytest.raises(TypeError):
        fid_func(real_images_tensor, 1)

    with pytest.raises(ValueError):
        fid_func(mindspore.ops.randint(0, 255, (1, 3, 255, 255)), fake_images_list)

    with pytest.raises(ValueError):
        fid_func(mindspore.ops.randint(0, 255, (3, 255, 255)), fake_images_list)

    with pytest.raises(ValueError):
        fid_func([Image.fromarray(real_image_np[0].transpose((1, 2, 0)).astype(np.uint8))], fake_images_tensor)

    with pytest.raises(TypeError):
        fid_func([mindspore.ops.randint(0, 255, (3, 255, 255)) for _ in range(10)], fake_images_list)


@pytest.mark.parametrize("feature", [64, 192, 768, 2048])
def test_fid_same_input(feature):
    """If real and fake are update on the same data the fid score should be 0."""
    metric = FrechetInceptionDistance(feature=feature)

    for _ in range(2):
        img = np.random.randint(0, 255, (10, 3, 299, 299))
        img = mindspore.Tensor(img).to(mindspore.uint8)
        metric.update(img, real=True)
        metric.update(img, real=False)

    val = metric.compute()
    assert mindspore.ops.isclose(val, mindspore.ops.zeros_like(val), atol=1e-5)


@pytest.mark.parametrize("feature_num", [64, 192, 768, 2048])
@pytest.mark.parametrize("normalize", [True, False])
def test_compare_fid(feature_num, normalize):
    """Check that the hole pipeline give the same result as torchmetrics."""

    np_tensor1 = np.random.randint(0, 200, (100, 3, 299, 299))
    np_tensor2 = np.random.randint(100, 255, (100, 3, 299, 299))

    # torchmetrics calculation
    imgs_dist1 = torch.from_numpy(np_tensor1).to(torch.uint8)
    imgs_dist2 = torch.from_numpy(np_tensor2).to(torch.uint8)

    fid_torch = TorchFrechetInceptionDistance(feature=feature_num, normalize=normalize)
    fid_torch.update(imgs_dist1, real=True)
    fid_torch.update(imgs_dist2, real=False)
    torch_fid_score = fid_torch.compute()
    print(torch_fid_score)

    # mindone calculation
    imgs_dist1_ms = mindspore.Tensor.from_numpy(np_tensor1).to(mindspore.uint8)
    imgs_dist2_ms = mindspore.Tensor.from_numpy(np_tensor2).to(mindspore.uint8)

    fid_ms = FrechetInceptionDistance(feature=feature_num, normalize=normalize)
    fid_ms.update(imgs_dist1_ms, real=True)
    fid_ms.update(imgs_dist2_ms, real=False)
    mindone_fid_score = fid_ms.compute()
    print(mindone_fid_score)

    # mindone functional calculation
    mindone_func_fid_score = fid_func(imgs_dist1_ms, imgs_dist2_ms, feature=feature_num, normalize=normalize)
    print(mindone_func_fid_score)

    print(np.allclose(torch_fid_score.numpy(), mindone_fid_score.asnumpy(), rtol=1e-5))
    print(np.allclose(torch_fid_score.numpy(), mindone_func_fid_score.asnumpy(), rtol=1e-5))


@pytest.mark.parametrize("reset_real_features", [True, False])
def test_reset_real_features_arg(reset_real_features):
    """Test that `reset_real_features` argument works as expected."""
    metric = FrechetInceptionDistance(feature=64, reset_real_features=reset_real_features)

    metric.update(mindspore.Tensor(np.random.randint(0, 180, (2, 3, 299, 299))).to(mindspore.uint8), real=True)
    metric.update(mindspore.Tensor(np.random.randint(0, 180, (2, 3, 299, 299))).to(mindspore.uint8), real=False)

    assert metric.real_features_num_samples == 2
    assert metric.real_features_sum.shape == (64,)
    assert metric.real_features_cov_sum.shape == (64, 64)

    assert metric.fake_features_num_samples == 2
    assert metric.fake_features_sum.shape == (64,)
    assert metric.fake_features_cov_sum.shape == (64, 64)

    metric.reset()

    # fake features should always reset
    assert metric.fake_features_num_samples == 0

    if reset_real_features:
        assert metric.real_features_num_samples == 0
    else:
        assert metric.real_features_num_samples == 2
        assert metric.real_features_sum.shape == (64,)
        assert metric.real_features_cov_sum.shape == (64, 64)


def test_not_enough_samples():
    """Test that an error is raised if not enough samples were provided."""
    img = np.random.randint(0, 255, (1, 3, 299, 299))
    img = mindspore.Tensor(img).to(mindspore.uint8)
    metric = FrechetInceptionDistance()
    metric.update(img, real=True)
    metric.update(img, real=False)
    with pytest.raises(
        RuntimeError, match="More than one sample is required for both the real and fake distributed to compute FID"
    ):
        metric.compute()
