import argparse
import os

import numpy as np
from PIL import Image
from tools._common import load_images
from tools.watermark.invisible_watermark.imwatermark import WatermarkDecoder, WatermarkEncoder

import mindspore as ms
from mindspore import Tensor, ops

# Copied from https://github.com/Stability-AI/generative-models/blob/613af104c6b85184091d42d374fef420eddb356d/scripts/demo/streamlit_helpers.py#L66
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
# WATERMARK_BITS='test'


class WatermarkEmbedder:
    def __init__(self, dtype=ms.float16):
        self.watermark = WATERMARK_BITS
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)
        self.dtype = dtype
        self.decoder = WatermarkDecoder("bits", len(self.watermark))

    def __call__(self, images):
        # can't encode images that are smaller than 256
        if images.shape[-1] < 256:
            return images
        images_ = []
        for i in range(images.shape[0]):
            im = ((255.0 * images[i].transpose(1, 2, 0))).astype(np.uint8).asnumpy()
            im = self.encoder.encode(im, "dwtDct")
            im = Tensor(np.asarray(im), self.dtype)
            images_.append(im)
        images = ops.stack(images_).transpose((0, 3, 1, 2))

        images = ops.clip_by_value((images / 255), clip_value_min=0.0, clip_value_max=1.0)

        return images

    def add_watermark(self, path):
        images, paths = load_images(path, resize=False)
        base_count = len(os.listdir(path))
        print(f"{len(images)} images are loaded")
        for image in images:
            im = np.array(image)
            im = self.encoder.encode(im, "dwtDct")
            images = Image.fromarray(im.astype(np.uint8))
            images.save(os.path.join(path, f"{base_count:05}.png"))
            base_count += 1

    def decoder_watermark(self, path):
        images, paths = load_images(path, resize=False)

        print(f"{len(images)} images are loaded")
        for image in images:
            im = np.array(image)
            im = self.decoder.decode(im, "dwtDct")
            print(im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Directory to images")
    # image_path = "/disk1/mindone/songyuanwei/mindone-master/examples/stable_diffusion_v2/output/samples_nowatermark/"
    args = parser.parse_args()
    watermark = WatermarkEmbedder()
    # images = watermark.add_watermark(args.image_path)
    images = watermark.decoder_watermark(args.image_path)
    print("done")
