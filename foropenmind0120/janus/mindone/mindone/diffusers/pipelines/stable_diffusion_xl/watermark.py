import numpy as np

import mindspore as ms
from mindspore import ops

from ...utils import is_invisible_watermark_available

if is_invisible_watermark_available():
    from imwatermark import WatermarkEncoder


# Adapted from https://github.com/Stability-AI/generative-models/blob/613af104c6b85184091d42d374fef420eddb356d/scripts/demo/streamlit_helpers.py#L66
# WATERMARK_MESSAGE: MindONE
WATERMARK_MESSAGE = 0b10110010100101101001000110011011101100001011000110111010
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]


class StableDiffusionXLWatermarker:
    def __init__(self):
        self.watermark = WATERMARK_BITS
        self.encoder = WatermarkEncoder()

        self.encoder.set_watermark("bits", self.watermark)

    def apply_watermark(self, images: ms.Tensor):
        # can't encode images that are smaller than 256
        if images.shape[-1] < 256:
            return images

        images = (255 * (images / 2 + 0.5)).permute(0, 2, 3, 1).float().numpy()

        # Convert RGB to BGR, which is the channel order expected by the watermark encoder.
        images = images[:, :, :, ::-1]

        # Add watermark and convert BGR back to RGB
        images = [self.encoder.encode(image, "dwtDct")[:, :, ::-1] for image in images]

        images = np.array(images)

        images = ms.Tensor.from_numpy(images).permute(0, 3, 1, 2)

        images = ops.clamp(2 * (images / 255 - 0.5), min=-1.0, max=1.0)
        return images
