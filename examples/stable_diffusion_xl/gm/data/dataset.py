import os
import random
import time

import imagesize
import numpy as np
import pandas as pd
from gm.util import instantiate_from_config
from PIL import Image


class Text2ImageDataset:
    dataset_column_names = ["samples"]

    def __init__(
        self,
        data_path,
        target_size=(1024, 1024),
        transforms=None,
        batched_transforms=None,
        tokenizer=None,
        image_filter_size=0,
        random_crop=False,
        filter_small_size=False,
        multi_aspect=None,  # for multi_aspect
        seed=42,  # for multi_aspect
        per_batch_size=1,  # for multi_aspect
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.target_size = [target_size, target_size] if isinstance(target_size, int) else target_size
        self.random_crop = random_crop
        self.filter_small_size = filter_small_size

        self.multi_aspect = list(multi_aspect) if multi_aspect is not None else None
        self.seed = seed
        self.per_batch_size = per_batch_size

        all_images, all_captions = self.list_image_files_captions_recursively(data_path)
        if filter_small_size:
            # print(f"Filter small images, filter size: {image_filter_size}")
            all_images, all_captions = self.filter_small_image(all_images, all_captions, image_filter_size)
        self.local_images = all_images
        self.local_captions = all_captions

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

        if self.tokenizer:
            raise NotImplementedError

    def __getitem__(self, idx):
        # images preprocess
        image_path = self.local_images[idx]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # caption preprocess
        caption = self.local_captions[idx]
        caption = caption if self.tokenizer is None else np.array(self.tokenize(caption), dtype=np.int32)
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

    def tokenize(self, text):
        SOT_TEXT = self.tokenizer.sot_text  # "[CLS]"
        EOT_TEXT = self.tokenizer.eot_text  # "[SEP]"
        CONTEXT_LEN = self.tokenizer.context_length

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        result = np.zeros([CONTEXT_LEN])
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]
        result[: len(tokens)] = tokens

        return result

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
        return {k: (np.stack(v, 0) if isinstance(v[0], np.ndarray) else v) for k, v in batch_samples.items()}

    def __len__(self):
        return len(self.local_images)

    @staticmethod
    def list_image_files_captions_recursively(data_path):
        anno_dir = data_path
        anno_list = sorted(
            [os.path.join(anno_dir, f) for f in list(filter(lambda x: x.endswith(".csv"), os.listdir(anno_dir)))]
        )
        db_list = [pd.read_csv(f) for f in anno_list]
        all_images = []
        all_captions = []
        for db in db_list:
            all_images.extend(list(db["dir"]))
            all_captions.extend(list(db["text"]))
        assert len(all_images) == len(all_captions)
        all_images = [os.path.join(data_path, f) for f in all_images]

        return all_images, all_captions

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
    parser.add_argument("--data_path", type=str)
    args, _ = parser.parse_known_args()
    transforms = [
        {"target": "gm.data.mappers.Resize", "params": {"size": 1024, "interpolation": 3}},
        {"target": "gm.data.mappers.Rescaler", "params": {"isfloat": False}},
        {"target": "gm.data.mappers.AddOriginalImageSizeAsTupleAndCropToSquare"},
    ]
    dataset = Text2ImageDataset(data_path=args.data_path, target_size=1024, transforms=transforms)
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
