from typing import Tuple

import numpy as np
import pytest
from opensora.datasets.transforms import Resize


@pytest.mark.parametrize(
    "input_size, target_size, output_size",
    [
        ((1080, 1920), (576, 1024), (576, 1024)),  # simple horizontal target for horizontal video
        ((1920, 1080), (1024, 576), (1024, 576)),  # simple vertical target for vertical video
        ((1080, 1920), (540, 1000), (562, 1000)),  # horizontal target for horizontal video
        ((1920, 1080), (1000, 540), (1000, 562)),  # vertical target for vertical video
        ((1080, 1920), (1024, 576), (1024, 1820)),  # vertical target for horizontal video
        ((1920, 1080), (576, 1024), (1820, 1024)),  # horizontal target for vertical video
        ((360, 640), (256, 640), (360, 640)),  # return itself
        ((640, 360), (640, 256), (640, 360)),  # return itself
        ((360, 640), (256, 720), (405, 720)),  # expand horizontally
        ((640, 360), (720, 256), (720, 405)),  # expand vertically
        ((360, 640), (720, 256), (720, 1280)),  # expand vertically horizontal video
        ((640, 360), (256, 720), (1280, 720)),  # expand horizontally vertical video
    ],
)
def test_resize_crop(input_size: Tuple[int, int], target_size: Tuple[int, int], output_size: Tuple[int, int]):
    image = np.random.randint(0, 256, (*input_size, 3), dtype=np.uint8)

    resize = Resize(target_size)
    output = resize(image)

    assert output.shape == (*output_size, 3)
