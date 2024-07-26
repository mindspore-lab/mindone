import copy
import csv
import glob
import logging
import os

import numpy as np
from opensora.utils.dataset_utils import create_image_transforms
from PIL import Image

import mindspore as ms

logger = logging.getLogger(__name__)


def get_image_path_list(folder):
    # TODO: find recursively
    fmts = ["jpg", "png", "jpeg", "JPEG"]
    out = []
    for fmt in fmts:
        out += glob.glob(os.path.join(folder, f"*.{fmt}"))
    return sorted(out)


class ImageDataset:
    def __init__(
        self,
        csv_path=None,
        data_folder=None,
        size=384,
        crop_size=256,
        random_crop=False,
        disable_flip=True,
        image_column="file_name",
        expand_dim_t=False,
    ):
        if csv_path is not None:
            with open(csv_path, "r") as csvfile:
                self.dataset = list(csv.DictReader(csvfile))
            self.read_from_csv = True
        else:
            self.dataset = get_image_path_list(data_folder)
            self.read_from_csv = False
        self.length = len(self.dataset)

        logger.info(f"Num data samples: {self.length}")

        self.data_folder = data_folder

        self.transform_backend = "al"  # pt, al
        self.pixel_transforms = create_image_transforms(
            size,
            crop_size,
            random_crop=random_crop,
            disable_flip=disable_flip,
            backend=self.transform_backend,
        )
        self.image_column = image_column
        self.expand_dim_t = expand_dim_t

        # prepare replacement data
        # max_attempts = 100
        # self.prev_ok_sample = self.get_replace_data(max_attempts)
        # self.require_update_prev = False

    def get_replace_data(self, max_attempts=100):
        replace_data = None
        attempts = min(max_attempts, self.length)
        for idx in range(attempts):
            try:
                pixel_values, caption = self.read_sample(idx)
                replace_data = copy.deepcopy((pixel_values, caption))
                break
            except Exception as e:
                print("\tError msg: {}".format(e), flush=True)

        assert replace_data is not None, f"Fail to preload sample in {attempts} attempts."

        return replace_data

    def read_sample(self, idx):
        if self.read_from_csv:
            image_dict = self.dataset[idx]
            # first column is image path
            image_fn = image_dict[list(image_dict.keys())[0]]
            image_path = os.path.join(self.data_folder, image_fn)
        else:
            image_path = self.dataset[idx]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        return image, image_path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # try:
        image, image_path = self.read_sample(idx)
        """
        if (self.prev_ok_sample is None) or (self.require_update_prev):
            self.prev_ok_sample = copy.deepcopy(image)
            self.require_update_prev = False
        except Exception as e:
            logger.warning(f"Fail to get sample of idx {idx}. The corrupted video will be replaced.")
            print("\tError msg: {}".format(e), flush=True)
            assert self.prev_ok_sample is not None
            image = self.prev_ok_sample  # unless the first sample is already not ok
            self.require_update_prev = True

            if idx >= self.length:
                raise IndexError  # needed for checking the end of dataset iteration
        """

        # import pdb
        # pdb.set_trace()

        if self.transform_backend == "pt":
            import torch

            pixel_values = torch.from_numpy(image).permute(2, 0, 1).contiguous()
            pixel_values = self.pixel_transforms(pixel_values)
            trans_image = pixel_values.numpy()
            out_image = (trans_image / 127.5 - 1.0).astype(np.float32)
        elif self.transform_backend == "al":
            trans_image = self.pixel_transforms(image=image)["image"]
            out_image = (trans_image / 127.5 - 1.0).astype(np.float32)
            out_image = out_image.transpose((2, 0, 1))  # h w c -> c h w

        if self.expand_dim_t:
            # c h w -> c t h w
            out_image = np.expand_dims(out_image, axis=1)

        return out_image, image_path


def check_sanity(x, save_fp="./tmp.png"):
    # reverse normalization and visulaize the transformed video
    if len(x.shape) == 4:
        print("only save the first image")
        x = x[0]
    x = np.transpose(x, (1, 2, 0))

    x = (x + 1.0) / 2.0  # -1,1 -> 0,1
    x = (x * 255).astype(np.uint8)

    if isinstance(x, ms.Tensor):
        x = x.asnumpy()
    Image.fromarray(x).save(save_fp)


if __name__ == "__main__":
    ds_config = dict(
        data_folder="/home/mindocr/yx/datasets/chinese_art_blip/train",
    )
    # test source dataset
    ds = ImageDataset(**ds_config)
    sample = ds.__getitem__(0)
    print(sample.shape)
