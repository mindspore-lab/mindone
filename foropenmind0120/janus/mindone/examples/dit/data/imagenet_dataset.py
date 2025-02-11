from typing import Optional

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore.dataset.transforms import Compose, vision

ALLOWED_FORMAT = {".jpeg", ".jpg", ".bmp", ".png"}


class _CenterCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Center cropping implementation from ADM.
        https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
        """
        while min(*img.size) >= 2 * self.size:
            img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)

        scale = self.size / min(*img.size)
        img = img.resize(tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC)

        arr = np.array(img)
        crop_y = (arr.shape[0] - self.size) // 2
        crop_x = (arr.shape[1] - self.size) // 2
        return Image.fromarray(arr[crop_y : crop_y + self.size, crop_x : crop_x + self.size])


def create_dataloader_imagenet(
    config,
    device_num: Optional[int] = None,
    rank_id: Optional[int] = None,
):
    dataset = ms.dataset.ImageFolderDataset(
        config["data_folder"],
        shuffle=config["shuffle"],
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=config["num_parallel_workers"],
        decode=False,
    )
    sample_size = config.get("sample_size", 256)
    dataset = dataset.map(
        operations=Compose(
            [
                vision.Decode(to_pil=True),
                _CenterCrop(sample_size),
                vision.RandomHorizontalFlip(),
                vision.HWC2CHW(),
                vision.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5], is_hwc=False),
            ]
        )
    )

    dl = dataset.batch(config["batch_size"], drop_remainder=True)
    return dl
