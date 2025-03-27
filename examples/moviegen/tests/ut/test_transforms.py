import cv2
import numpy as np
from mg.dataset.transforms import HorizontalFlip, ResizeCrop


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


def test_horizontal_video_crop():
    video = np.random.randint(0, 256, (10, 150, 250, 3), dtype=np.uint8)
    rc = ResizeCrop((100, 200))
    video = rc(video)
    assert video.shape == (10, 100, 200, 3)


def test_vertical_video_crop():
    video = np.random.randint(0, 256, (10, 250, 150, 3), dtype=np.uint8)
    rc = ResizeCrop((100, 200))
    video = rc(video)
    assert video.shape == (10, 200, 100, 3)


def test_horizontal_flip_image():
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    flipped_image = HorizontalFlip(p=1)(image)
    assert (flipped_image == cv2.flip(image, 1)).all()


def test_horizontal_flip_video():
    video = np.random.randint(0, 256, (10, 128, 128, 3), dtype=np.uint8)
    flipped_video = HorizontalFlip(p=1)(video)
    assert (flipped_video == np.stack([cv2.flip(f, 1) for f in video])).all()
