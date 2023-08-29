import random

from PIL import Image

import mindspore as ms
from mindspore import ops
from mindspore.dataset import vision
from mindspore.dataset.vision import Inter as InterpolationMode

__all__ = [
    "RandomResize",
    "CenterCrop",
    "AddGaussianNoise",
    "make_masked_images",
]


class RandomResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = vision.Resize(
            self.size,
            interpolation=random.choice(
                [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC, InterpolationMode.ANTIALIAS]
            ),
        )(img)
        return img


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # fast resize
        while min(img.size) >= 2 * self.size:
            img = img.resize((img.width // 2, img.height // 2), resample=Image.BOX)
        scale = self.size / min(img.size)
        img = img.resize((round(scale * img.width), round(scale * img.height)), resample=Image.BICUBIC)

        # center crop
        x1 = (img.width - self.size) // 2
        y1 = (img.height - self.size) // 2
        img = img.crop((x1, y1, x1 + self.size, y1 + self.size))
        return img


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        assert isinstance(img, ms.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(ms.float32)
        out = img + self.std * ops.randn_like(img) + self.mean
        if out.dtype != dtype:
            out = out.to(dtype)
        return out

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


def make_masked_images(imgs, masks):
    masked_imgs = []
    for i, mask in enumerate(masks):
        # concatenation
        masked_imgs.append(ops.cat([imgs[i] * (1 - mask), (1 - mask)], axis=1))
    return ops.stack(masked_imgs, axis=0)
