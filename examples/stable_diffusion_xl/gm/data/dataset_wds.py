import copy
import glob
import io
import json
import os
import random
import time

import numpy as np
import webdataset as wds
import wids
from gm.util import instantiate_from_config
from PIL import Image


def get_tar_file_list(data_dir):
    # get tar file recursively
    tar_files = []
    tar_files.extend(glob.glob(os.path.join(data_dir, "*.tar")))

    folders = [fp for fp in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, fp))]
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        tar_files.extend(get_tar_file_list(folder_path))

    return tar_files


def get_tar_nsample(tar_file):
    # TODO: improve efficiency.
    wds_iterator = wds.WebDataset(tar_file)
    n = 0
    for cur in wds_iterator:
        n += 1
    return n


def generate_sharlist(data_dir):
    tar_files = get_tar_file_list(data_dir)
    out = {
        "__kind__": "wids-shard-index-v1",
        "wids_version": 1,
        "shardlist": [],
    }
    for tf in tar_files:
        nsamples = get_tar_nsample(tf)
        out["shardlist"].append({"url": tf, "nsamples": nsamples})
    save_fp = os.path.join(data_dir, "data_info.json")
    with open(save_fp, "w") as fp:
        json.dump(out, fp)

    return save_fp


class T2I_BaseDataset:
    def __init__(
        self,
        data_path,
        num_samples=None,  # need for webdataset to get data len
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
        caption_key="caption",
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
        assert not filter_small_size, "filter small size is not supported"

        self.multi_aspect = list(multi_aspect) if multi_aspect is not None else None
        self.seed = seed
        self.per_batch_size = per_batch_size

        self.caption_key = caption_key
        self.prev_ok_sample = None
        self.require_update_prev = True

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

    def preprocess(self, image, caption: str):
        # preprocess image and caption
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

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


class T2I_Webdataset(T2I_BaseDataset):
    # sequential reading
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data_path = kwargs.get("data_path")
        num_samples = kwargs.get("num_samples")

        tar_files = get_tar_file_list(data_path)
        print(f"Get {len(tar_files)} tar files")

        self.wds_iterator = wds.WebDataset(tar_files, cache_dir=None)
        self.wds_iterator = self.wds_iterator.shuffle(1000)
        # ds = ds.decode("rgb8").to_tuple("jpg;png", "json") # will do in getitem to save time
        if num_samples is None:
            print(
                "WARNING: For webdataset, it's recommended to specify `num_samples` to save time to iterate all samples for counting"
            )
            self.num_samples = self.count_sample_num(self.wds_iterator)
            print(f"Total number of samples: {self.num_samples} in all tar files")
        else:
            self.num_samples = num_samples

    def parse_raw_data(self, raw_data):
        if "jpg" in raw_data:
            image = Image.open(io.BytesIO(raw_data["jpg"]))
        else:
            image = Image.open(io.BytesIO(raw_data["png"]))

        annot = json.load(io.BytesIO(raw_data["json"]))
        if self.caption_key in annot:
            caption = annot[self.caption_key]
        else:
            raise ValueError("No caption found. Expecting caption key: {}".self.caption_key)

        return image, caption

    def __iter__(self):
        # images preprocess
        for raw in self.wds_iterator:
            try:
                image, caption = self.parse_raw_data(raw)
                sample = self.preprocess(image, caption)
                # TODO: add corrupted data check and replacement
                yield sample
            except StopIteration:
                raise StopIteration


class T2I_Webdataset_RndAcs(T2I_BaseDataset):
    # random access
    def __init__(self, shardlist_desc=None, *args, **kwargs):
        # shardlist_desc: path to a json file describing sample num for each tar
        super().__init__(*args, **kwargs)
        if shardlist_desc is None:
            data_path = kwargs.get("data_path")
            if not os.path.exists(os.path.join(data_path, "data_info.json")):
                print("Scanning tar files to get sample nums...")
                shardlist_desc = generate_sharlist(data_path)
                print("=> Saved shardlist json file in ", shardlist_desc)
            else:
                shardlist_desc = os.path.join(data_path, "data_info.json")
        print("Loading sharlist description from: ", shardlist_desc)

        with open(shardlist_desc, "r") as fp:
            shardlist = json.load(fp)["shardlist"]
        self.dataset = wids.ShardListDataset(shardlist)
        self._datalen = len(self.dataset)

    def parse_raw_data(self, raw_data):
        # parse webdataset reading result
        if ".jpg" in raw_data:
            image = raw_data[".jpg"]
        elif "png" in raw_data:
            image = raw_data[".png"]
        else:
            raise ValueError("Missing jpg/png image, only get keys: {}".format(raw_data.keys()))

        annot = raw_data[".json"]
        if self.caption_key in annot:
            caption = annot[self.caption_key]
        else:
            raise ValueError("No caption found. Expecting caption key: {}".format(self.caption_key))

        return image, caption

    def __getitem__(self, idx):
        try:
            raw = self.dataset[idx]
            image, caption = self.parse_raw_data(raw)
            sample = self.preprocess(image, caption)
            if (self.prev_ok_sample is None) or (self.require_update_prev):
                self.prev_ok_sample = copy.deepcopy(sample)
                self.require_update_prev = False
        except Exception as e:
            print(
                f"=> WARNING: Fail to get sample {idx}. The sample can be corrupted and will be replaced by previous normal sample."
            )
            print("\tError type: ", type(e).__name__)
            print("\tError mg: {}".format(e), flush=True)
            sample = self.prev_ok_sample  # unless the first sample is already not ok
            self.require_update_prev = True

            if idx >= self._datalen:
                raise IndexError  # needed for checking the end of dataset iteration

        return sample

    def __len__(self):
        return self._datalen


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="check dataset")
    parser.add_argument("--target", type=str, default="T2I_Webdataset_RndAcs")
    # for Text2ImageDataset
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--shardlist_desc", type=str, default=None)
    parser.add_argument("--caption_key", type=str, default="caption")
    # for Text2ImageDatasetDreamBooth
    args, _ = parser.parse_known_args()
    transforms = [
        {"target": "gm.data.mappers.Resize", "params": {"size": 1024, "interpolation": 3}},
        {"target": "gm.data.mappers.Rescaler", "params": {"isfloat": False}},
        {"target": "gm.data.mappers.AddOriginalImageSizeAsTupleAndCropToSquare"},
    ]
    print("loading..")
    if args.target == "T2I_Webdataset":
        dataset = T2I_Webdataset(
            data_path=args.data_path, target_size=1024, transforms=transforms, caption_key=args.caption_key
        )
    elif args.target == "T2I_Webdataset_RndAcs":
        dataset = T2I_Webdataset_RndAcs(
            data_path=args.data_path,
            shardlist_desc=args.shardlist_desc,
            target_size=1024,
            transforms=transforms,
            caption_key=args.caption_key,
        )
    else:
        raise ValueError("Unknown dataset target")

    dataset_size = len(dataset)
    print(f"dataset size: {dataset_size}")

    s_time = time.time()
    tot_time = 0
    n_read = len(dataset)
    for i, data in enumerate(dataset):
        if i > n_read:
            break
        tot_time += time.time() - s_time
        # print(f"{i}/{dataset_size}, image shape: {data.pop('image')}, {data}")
        print(f"{i+1}/{dataset_size}, time cost: {(time.time()-s_time) * 1000} ms")
        print(data["txt"])
        s_time = time.time()
    print("Total read time: ", tot_time)
