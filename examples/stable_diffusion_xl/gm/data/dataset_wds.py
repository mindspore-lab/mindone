import os
import random
import time
import glob

import imagesize
import numpy as np
import pandas as pd
from PIL import Image
import webdataset as wds
from itertools import islice
import io
import json

from gm.util import instantiate_from_config


def get_tar_file_list(data_dir):
    # get tar file recursively
    tar_files = []
    # two levesl
    tar_files.extend(glob.glob(os.path.join(data_dir, '*.tar')))

    folders = [fp for fp in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, fp))]
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        tar_files.extend(get_tar_file_list(folder_path))

    return tar_files


class T2I_RandWebdataset:
    def __init__(
        self,
        data_path,
        num_samples=None, # required for webdataset. If None, it will spend time to count
        target_size=(1024, 1024),
        transforms=None,
        batched_transforms=None,
        tokenizer=None,
        token_nums=None,
        image_filter_size=0,
        random_crop=False,
        filter_small_size=False,
        multi_aspect=None,  # for multi_aspect
        seed=42,  # for multi_aspect
        per_batch_size=1,  # for multi_aspect
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_nums = token_nums
        self.dataset_column_names = ["samples"]
        if self.tokenizer is None:
            self.dataset_output_column_names = self.dataset_column_names
        else:
            assert token_nums is not None and token_nums > 0
            self.dataset_output_column_names = [
                "image",
            ] + [f"token{i}" for i in range(token_nums)]

        self.target_size = [target_size, target_size] if isinstance(target_size, int) else target_size
        self.random_crop = random_crop
        self.filter_small_size = filter_small_size

        self.multi_aspect = list(multi_aspect) if multi_aspect is not None else None
        self.seed = seed
        self.per_batch_size = per_batch_size

        # create webdataset iterator for tar files
        tar_files = get_tar_file_list(data_path)
        print(f"Get {len(tar_files)} tar files")
        self.wds_iterator = wds.WebDataset(tar_files, cache_dir=None)
        if num_samples is None:
            print("WARNING: For webdataset, it's recommended to specify `num_samples` to save time to iterate all samples for counting")
            self.num_samples = self.count_sample_num(self.wds_iterator)
            print(f"Total number of samples: {self.num_samples} in all tar files")
        else:
            self.num_samples = num_samples

        # ds = ds.shuffle(1000) # uncomment for local shuffle. no needed if shuffle in generator dataset
        # ds = ds.decode("rgb8").to_tuple("jpg;png", "json") # will do in getitem to save time

        self.transforms = []
        if transforms:
            for i, trans_config in enumerate(transforms):
                # Mapper
                trans = instantiate_from_config(trans_config)
                self.transforms.append(trans)
                print(f"Adding mapper {trans.__class__.__name__} as transform #{i} " f"to the datapipeline")

        self.batched_transforms = []
        if batched_transforms:
            for i, bs_trans_config in enumerate(batched_transforms):
                # Mapper
                bs_trans = instantiate_from_config(bs_trans_config)
                self.batched_transforms.append(bs_trans)
                print(
                    f"Adding batch mapper {bs_trans.__class__.__name__} as batch transform #{i} " f"to the datapipeline"
                )

    def __getitem__(self, idx):
        # images preprocess
        '''
        image_path = self.local_images[idx]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        '''
        raw = next(islice(self.wds_iterator, idx, idx+1))
        image = Image.open(io.BytesIO(raw['jpg']))
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        annot = json.load(io.BytesIO(raw['json']))
        caption = annot['caption']

        # caption preprocess
        caption = np.array(caption)

        sample = {
            "image": image,
            "txt": caption,
            "original_size_as_tuple": np.array([image.shape[0], image.shape[1]]),  # original h, original w
            "target_size_as_tuple": np.array([self.target_size[0], self.target_size[1]]),  # target h, target w
            "crop_coords_top_left": np.array([0, 0]),  # crop top, crop left
            "aesthetic_score": np.array(
                [
                    6.0,
                ]
            ),
        }

        for trans in self.transforms:
            sample = trans(sample)

        return sample

    def collate_fn(self, samples, batch_info):
        new_size = self.target_size
        if self.multi_aspect:
            epoch_num, batch_num = batch_info.get_epoch_num(), batch_info.get_batch_num()
            cur_seed = epoch_num * 10 + batch_num
            random.seed(cur_seed)
            new_size = random.choice(self.multi_aspect)

        for bs_trans in self.batched_transforms:
            samples = bs_trans(samples, target_size=new_size)

        batch_samples = {k: [] for k in samples[0]}
        for s in samples:
            for k in s:
                batch_samples[k].append(s[k])

        data = {k: (np.stack(v, 0) if isinstance(v[0], np.ndarray) else v) for k, v in batch_samples.items()}

        if self.tokenizer:
            data = {k: (v.tolist() if k == "txt" else v.astype(np.float32)) for k, v in data.items()}
            tokens, _ = self.tokenizer(data)
            outs = (data["image"],) + tuple(tokens)
        else:
            outs = data

        return outs

    def __len__(self):
        return self.num_samples

    @staticmethod
    def count_sample_num(wds_iterator):
        cnt = 0
        for cur in wds_iterator:
            cnt += 1

        return cnt

    @staticmethod
    def filter_small_image(all_images, all_captions, image_filter_size):
        filted_images = []
        filted_captions = []
        for image, caption in zip(all_images, all_captions):
            w, h = imagesize.get(image)
            if min(w, h) < image_filter_size:
                print(f"The size of image {image}: {w}x{h} < `image_filter_size` and excluded from training.")
                continue
            else:
                filted_images.append(image)
                filted_captions.append(caption)
        return filted_images, filted_captions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="check dataset")
    parser.add_argument("--target", type=str, default="T2I_RandWebdataset")
    # for Text2ImageDataset
    parser.add_argument("--data_path", type=str, default="")
    # for Text2ImageDatasetDreamBooth
    args, _ = parser.parse_known_args()
    transforms = [
        {"target": "gm.data.mappers.Resize", "params": {"size": 1024, "interpolation": 3}},
        {"target": "gm.data.mappers.Rescaler", "params": {"isfloat": False}},
        {"target": "gm.data.mappers.AddOriginalImageSizeAsTupleAndCropToSquare"},
    ]
    print('loading..')
    if args.target == "T2I_RandWebdataset":
        dataset = T2I_RandWebdataset(data_path=args.data_path, target_size=1024, transforms=transforms)
        dataset_size = len(dataset)
        print(f"dataset size: {dataset_size}")

        s_time = time.time()
        for i, data in enumerate(dataset):
            if i > 9:
                break
            print(
                f"{i}/{dataset_size}, image shape: {data.pop('image')}, {data}, "
                f"time cost: {(time.time()-s_time) * 1000} ms"
            )
            s_time = time.time()
    else:
        raise ValueError("Unknown dataset target")
