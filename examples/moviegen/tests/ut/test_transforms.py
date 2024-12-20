import numpy as np
from mg.dataset.transforms import ResizeCrop


def test_horizontal_image_crop():
    image = np.random.randint(0, 256, (150, 250, 3), dtype=np.uint8)
    rc = ResizeCrop((100, 200))
    image = rc(image)
    assert image.shape == (100, 200, 3)


def test_vertical_image_crop():
    image = np.random.randint(0, 256, (250, 150, 3), dtype=np.uint8)
    rc = ResizeCrop((100, 200))
    image = rc(image)
    assert image.shape == (200, 100, 3)
