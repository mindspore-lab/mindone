import os
import random
import time

import imagesize
import numpy as np
import pandas as pd
from gm.util import instantiate_from_config
from PIL import Image


class Text2ImageDataset:
    def __init__(
        self,
        data_path,
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

    def __getitem__(self, idx):
        # images preprocess
        image_path = self.local_images[idx]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # caption preprocess
        caption = self.local_captions[idx]
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


class Text2ImageDatasetDreamBooth:
    dataset_column_names = ["instance_samples", "class_samples"]

    def __init__(
        self,
        instance_data_path,
        class_data_path,
        instance_prompt,
        class_prompt,
        train_data_repeat=1,
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

        instance_images = self.list_image_files_recursively(instance_data_path)
        instance_images = self.repeat_data(instance_images, train_data_repeat)
        print(
            f"The training data is repeated {train_data_repeat} times, and the total number is {len(instance_images)}"
        )

        class_images = self.list_image_files_recursively(class_data_path)
        if filter_small_size:
            instance_images = self.filter_small_image(instance_images, image_filter_size)
            class_images = self.filter_small_image(class_images, image_filter_size)

        self.instance_images = instance_images
        self.class_images = class_images
        self.instance_caption = instance_prompt
        self.class_caption = class_prompt

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
        instance_image_path = self.instance_images[idx]
        instance_image = Image.open(instance_image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = np.array(instance_image).astype(np.uint8)

        class_image_path = random.choice(self.class_images)
        class_image = Image.open(class_image_path)
        if not class_image.mode == "RGB":
            class_image = class_image.convert("RGB")
        class_image = np.array(class_image).astype(np.uint8)

        # caption preprocess
        instance_caption = (
            self.instance_caption
            if self.tokenizer is None
            else np.array(self.tokenize(self.instance_caption), dtype=np.int32)
        )
        instance_caption = np.array(instance_caption)

        class_caption = (
            self.class_caption
            if self.tokenizer is None
            else np.array(self.tokenize(self.class_caption), dtype=np.int32)
        )
        class_caption = np.array(class_caption)

        instance_sample = {
            "image": instance_image,
            "txt": instance_caption,
            "original_size_as_tuple": np.array(
                [instance_image.shape[0], instance_image.shape[1]]
            ),  # original h, original w
            "target_size_as_tuple": np.array([self.target_size[0], self.target_size[1]]),  # target h, target w
            "crop_coords_top_left": np.array([0, 0]),  # crop top, crop left
            "aesthetic_score": np.array(
                [
                    6.0,
                ]
            ),
        }

        class_sample = {
            "image": class_image,
            "txt": class_caption,
            "original_size_as_tuple": np.array([class_image.shape[0], class_image.shape[1]]),  # original h, original w
            "target_size_as_tuple": np.array([self.target_size[0], self.target_size[1]]),  # target h, target w
            "crop_coords_top_left": np.array([0, 0]),  # crop top, crop left
            "aesthetic_score": np.array(
                [
                    6.0,
                ]
            ),
        }

        for trans in self.transforms:
            instance_sample = trans(instance_sample)
            class_sample = trans(class_sample)

        return instance_sample, class_sample

    def collate_fn(self, instance_samples, class_samples, batch_info):
        new_size = self.target_size
        if self.multi_aspect:
            epoch_num, batch_num = batch_info.get_epoch_num(), batch_info.get_batch_num()
            cur_seed = epoch_num * 10 + batch_num
            random.seed(cur_seed)
            new_size = random.choice(self.multi_aspect)

        for bs_trans in self.batched_transforms:
            instance_samples = bs_trans(instance_samples, target_size=new_size)
            class_samples = bs_trans(class_samples, target_size=new_size)

        instance_batch_samples = {k: [] for k in instance_samples[0]}
        class_batch_samples = {k: [] for k in class_samples[0]}
        for s in instance_samples:
            for k in s:
                instance_batch_samples[k].append(s[k])
        for s in class_samples:
            for k in s:
                class_batch_samples[k].append(s[k])
        instance_batch_samples = {
            k: (np.stack(v, 0) if isinstance(v[0], np.ndarray) else v) for k, v in instance_batch_samples.items()
        }
        class_batch_samples = {
            k: (np.stack(v, 0) if isinstance(v[0], np.ndarray) else v) for k, v in class_batch_samples.items()
        }
        return instance_batch_samples, class_batch_samples

    def __len__(self):
        return len(self.instance_images)

    @staticmethod
    def list_image_files_recursively(image_path):
        image_path_list = sorted(os.listdir(image_path))
        all_images = [os.path.join(image_path, f) for f in image_path_list]
        return all_images

    @staticmethod
    def filter_small_image(all_images, image_filter_size):
        filted_images = []
        for image in all_images:
            w, h = imagesize.get(image)
            if min(w, h) < image_filter_size:
                print(f"The size of image {image}: {w}x{h} < `image_filter_size` and excluded from training.")
                continue
            else:
                filted_images.append(image)
        return filted_images

    @staticmethod
    def repeat_data(data_list, repeats):
        return data_list * repeats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="check dataset")
    parser.add_argument("--target", type=str, default="Text2ImageDataset")
    # for Text2ImageDataset
    parser.add_argument("--data_path", type=str, default="")
    # for Text2ImageDatasetDreamBooth
    parser.add_argument("--instance_data_path", type=str, default="")
    parser.add_argument("--class_data_path", type=str, default="")
    parser.add_argument("--instance_prompt", type=str, default="")
    parser.add_argument("--class_prompt", type=str, default="")
    args, _ = parser.parse_known_args()
    transforms = [
        {"target": "gm.data.mappers.Resize", "params": {"size": 1024, "interpolation": 3}},
        {"target": "gm.data.mappers.Rescaler", "params": {"isfloat": False}},
        {"target": "gm.data.mappers.AddOriginalImageSizeAsTupleAndCropToSquare"},
    ]

    if args.target == "Text2ImageDataset":
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

    elif args.target == "Text2ImageDatasetDreamBooth":
        dataset = Text2ImageDatasetDreamBooth(
            instance_data_path=args.instance_data_path,
            class_data_path=args.class_data_path,
            instance_prompt=args.instance_prompt,
            class_prompt=args.class_prompt,
            target_size=1024,
            transforms=transforms,
        )
        dataset_size = len(dataset)
        print(f"dataset size: {dataset_size}")

        for i, data in enumerate(dataset):
            print(data)
            break

    else:
        ValueError("dataset only support Text2ImageDataset and Text2ImageDatasetDreamBooth")
