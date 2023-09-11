import random

from PIL import Image

import mindspore as ms
from mindspore import ops
from mindspore.dataset import vision, transforms 
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


# TODO: add augmentation
def create_transforms(cfg, is_training=True):
    # [Transform] Transforms for different inputs

    # video frames, norm to [-1, 1] for VAE
    infer_transforms = transforms.Compose(
        [
            vision.CenterCrop(size=cfg.resolution),
            vision.ToTensor(),
            vision.Normalize(mean=cfg.mean, std=cfg.std, is_hwc=False),
        ]
    )
    # NOTE: only norm to [0. 1] for stc encoder or for detph/sketch image preprocessor
    misc_transforms = transforms.Compose(
        [
            RandomResize(size=cfg.misc_size),
            vision.CenterCrop(cfg.misc_size),
            vision.ToTensor(),
        ]
    )
    # since is motion data, no norm
    mv_transforms = transforms.Compose(
        [
            vision.Resize(size=cfg.resolution),
            vision.CenterCrop(cfg.resolution),
        ]
    )
    # 
    vit_transforms = transforms.Compose(
        [
            CenterCrop(cfg.vit_image_size),
            vision.ToTensor(), # to chw
            vision.Normalize(mean=cfg.vit_mean, std=cfg.vit_std, is_hwc=False),
        ]
    )
    
    # depth/motion net transforms
    # depth_input_process = ...

    return infer_transforms, misc_transforms, mv_transforms, vit_transforms


