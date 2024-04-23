import copy
import glob
import io
import json
import math
import os
import random
import time
from functools import partial
from itertools import islice

import numpy as np
import webdataset as wds
import wids
from gm.data.util import _is_valid_text_input
from gm.util import instantiate_from_config
from PIL import Image
from tqdm import tqdm

from mindspore.communication import get_group_size, get_rank


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
    print("INFO: Start to scan tar files...")
    # TODO: 1) use multi-process. 2) consider multiple machine access.
    for tf in tqdm(tar_files):
        tar_info_fp = tf.replace(".tar", ".txt")
        if not os.path.exists(tar_info_fp):
            # scan
            nsamples = get_tar_nsample(tf)

            with open(tar_info_fp, "w") as fp:
                fp.write(str(nsamples))
        else:
            with open(tar_info_fp, "r") as fp:
                nsamples = int(fp.read())

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
        prompt_empty_probability=0.0,
        lpw=False,
        max_embeddings_multiples=4,
        return_sample_name=False,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(
                "WARNING: Some key arguments are fed but not supported in the T2I_BaseDataset"
                " ".join(list(kwargs.keys()))
            )

        self.tokenizer = tokenizer
        self.token_nums = token_nums
        self.dataset_column_names = ["samples"]
        self.return_sample_name = return_sample_name
        self.data_path = data_path

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
        self.prompt_empty_probability = prompt_empty_probability

        self.caption_key = caption_key
        self.prev_ok_sample = None
        self.require_update_prev = True
        self.lpw = lpw
        self.max_embeddings_multiples = max_embeddings_multiples

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

    def preprocess(self, image, caption: str, image_path="0000000000000"):
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # caption preprocess
        if self.prompt_empty_probability and random.random() < self.prompt_empty_probability:
            caption = ""

        if not _is_valid_text_input(caption):
            print(
                f"WARNING: text input must of type `str`, but got type: {type(caption)}, caption: {caption}", flush=True
            )

            caption = str(caption)

            if _is_valid_text_input(caption):
                print("WARNING: convert caption type to string success.", flush=True)
            else:
                caption = " "
                print("WARNING: convert caption type to string fail, set caption to ` `.", flush=True)
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

        if self.return_sample_name:
            self.dataset_output_column_names.append("sample_name")
            sample["sample_name"] = np.array(image_path)

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
            data = {
                k: (v.tolist() if (k == "txt" or k == "sample_name") else v.astype(np.float32)) for k, v in data.items()
            }
            try:
                tokens, _ = self.tokenizer(data, lpw=self.lpw, max_embeddings_multiples=self.max_embeddings_multiples)
            except Exception as e:
                print(f"WARNING: tokenize fail, error mg: {e}, convert data[`txt`]: {data['txt']} to ` `", flush=True)
                data["txt"] = [" " for _ in range(len(data["txt"]))]
                tokens, _ = self.tokenizer(data, lpw=self.lpw, max_embeddings_multiples=self.max_embeddings_multiples)
            outs = (data["image"],) + tuple(tokens)
            if "sample_name" in data:
                outs += (data["sample_name"],)
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
            # print(cnt)

        return cnt


def get_device_rank_info():
    # device_id = int(os.getenv("DEVICE_ID", 0))
    try:
        rank_id = get_rank()
        device_num = get_group_size()
    except Exception:
        # print(
        #     "WARNING: Distributed Communication has not been inited (by init()). rank_id and rank_size will be retrieved from env variables."
        # )
        rank_id = int(os.environ.get("RANK_ID", 0))
        device_num = int(os.environ.get("RANK_SIZE", 1))

    # print(f"D--: device_num: {device_num}, rank_id {rank_id}")

    return rank_id, device_num


def split_by_node(src, group=None, rank_id=None, rank_size=None):
    assert group is None, "currently only support group is None"
    rank, world_size = get_device_rank_info()
    if rank_id is not None:
        rank = rank_id
    if rank_size is not None:
        world_size = rank_size

    if world_size > 1:
        yield from islice(src, rank, None, world_size)
    else:
        yield from src


def split_by_worker(src):
    # Split the input sequence by worker.
    # rank, world_size, worker, num_workers = utils.pytorch_worker_info()
    worker = 0
    num_workers = 1
    if num_workers > 1:
        yield from islice(src, worker, None, num_workers)
    else:
        yield from src


def get_num_samples(shardlist_desc=None, data_path=None):
    # data_path: root dir of tar dataset
    if shardlist_desc is None:
        assert data_path is not None
        if not os.path.exists(os.path.join(data_path, "data_info.json")):
            shardlist_desc_file = os.path.join(data_path, "data_info.json")
            raise FileNotFoundError(
                f"{shardlist_desc_file} not found, please prepare dataset meta info before training, "
                f"generate through `tools/data_check/get_wds_num_samples.py`"
            )
        else:
            shardlist_desc = os.path.join(data_path, "data_info.json")
    print("Loading sharlist description from: ", shardlist_desc, flush=True)

    tot_samples = 0
    with open(shardlist_desc, "r") as fp:
        shardlist = json.load(fp)["shardlist"]
        for shard in shardlist:
            tot_samples += shard["nsamples"]

    return tot_samples


class T2I_Webdataset(T2I_BaseDataset):
    """
    Webdataset loading, support data sharding for multiple training nodes.
    """

    def __init__(self, shardlist_desc=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data_path = kwargs.get("data_path")
        num_samples = kwargs.get("num_samples", -1)

        tar_files = get_tar_file_list(data_path)
        print(f"Get {len(tar_files)} tar files")

        # get number of samples
        if num_samples == -1:
            tot_samples = get_num_samples(shardlist_desc, data_path)
        else:
            tot_samples = num_samples

        # Change the epoch to return the given number of samples, determine by total samples and rank
        rank_id, device_num = get_device_rank_info()
        samples_per_rank = math.ceil(tot_samples / device_num)
        print(
            f"INFO: Total samples in dataset {tot_samples}, device num {device_num}, "
            f"rank id {rank_id}, num samples per device: {samples_per_rank}"
        )

        if device_num > len(tar_files):
            print(
                f"WARNING: RankSize {device_num} greater than WebDataset tar files num {len(tar_files)}, "
                f"tar files will be sampled repeatedly.",
            )
            device_num = len(tar_files)
            rank_id %= device_num

        # webdataset with shard split
        # self.wds_iterator = wds.WebDataset(tar_files, resampled=True, cache_dir=cache_dir, nodesplitter=split_by_node)
        self.wds_iterator = wds.WebDataset(
            tar_files,
            cache_dir=None,
            nodesplitter=partial(split_by_node, rank_id=rank_id, rank_size=device_num),
            workersplitter=split_by_worker,
        )
        self.wds_iterator = self.wds_iterator.with_epoch(samples_per_rank)
        self.num_samples = samples_per_rank

        self.wds_iterator = self.wds_iterator.shuffle(1000)  # TODO: allow set shuffle window size
        # ds = ds.decode("rgb8").to_tuple("jpg;png", "json") # will do in getitem to save time

        # prepare normal sample for replacement
        max_attempts = 100
        trials = 0
        for raw in self.wds_iterator:
            try:
                image, caption = self.parse_raw_data(raw)
                if "__key__" in raw:
                    sample = self.preprocess(image, caption, str(raw["__key__"]))
                else:
                    print("=> WARNING: Fail to get the attribute __key__. using white space instead")
                    sample = self.preprocess(image, caption, " ")
                trials += 1
                if sample is not None:
                    self.prev_ok_sample = copy.deepcopy(sample)
                    break
                assert trials < max_attempts, f"Cannot get normal samples in {max_attempts} attempts"
            except StopIteration:
                raise StopIteration
            except Exception as e:
                print("\tError mg: {}".format(e), flush=True)
                continue

        print(f"Finish preparing normal sample in {trials} attempt(s)")

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
                if "__key__" in raw:
                    sample = self.preprocess(image, caption, str(raw["__key__"]))
                else:
                    print("=> WARNING: Fail to get the attribute __key__. using white space instead")
                    sample = self.preprocess(image, caption, " ")
                yield sample
            except StopIteration:
                raise StopIteration
            except Exception as e:
                # Print damaged samples
                caption = None
                try:
                    if "json" in raw and self.caption_key:
                        annot = json.load(io.BytesIO(raw["json"]))
                        if self.caption_key in annot:
                            caption = annot[self.caption_key]
                        else:
                            raise ValueError(f"No caption found. Expecting caption key: {self.caption_key}")
                except Exception:
                    pass
                if caption:
                    print(f"\tDamaged samples, caption: {caption}, raw data: {raw}")
                else:
                    print(f"\tDamaged samples, load caption fail, raw data: {raw}")
                ####################

                print(
                    "=> WARNING: Fail to get the iterated sample. The sample can be corrupted and will be replaced by previous normal sample."
                )
                print("\tError type: ", type(e).__name__)
                print("\tError mg: {}".format(e), flush=True)
                assert self.prev_ok_sample is not None
                sample = self.prev_ok_sample  # unless the first sample is already not ok
                self.require_update_prev = True

                yield sample


class T2I_Webdataset_RndAcs(T2I_BaseDataset):
    # random access
    def __init__(self, shardlist_desc=None, cache_dir=None, *args, **kwargs):
        # shardlist_desc: path to a json file describing sample num for each tar
        super().__init__(*args, **kwargs)
        self.data_path = kwargs.get("data_path")
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
        self.dataset = wids.ShardListDataset(shardlist, cache_dir=cache_dir)
        self._datalen = len(self.dataset)

        # preload sample
        for _ in range(100):
            try:
                _idx = random.randint(0, self._datalen - 1)
                raw = self.dataset[_idx]
                image, caption = self.parse_raw_data(raw)
                sample = self.preprocess(image, caption)
            except Exception as e:
                print(
                    f"=> WARNING: Fail to preload sample {_idx}. "
                    f"The sample can be corrupted and will be replaced by previous normal sample."
                )
                print("\tError type: ", type(e).__name__)
                print("\tError mg: {}".format(e), flush=True)
                continue

            if sample is not None:
                self.prev_ok_sample = copy.deepcopy(sample)
                break

        assert self.prev_ok_sample is not None, "=> Error: Fail to preload sample."

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
            assert self.prev_ok_sample is not None
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
        print(f"{i + 1}/{dataset_size}, time cost: {(time.time() - s_time) * 1000} ms")
        print(data["txt"])
        s_time = time.time()
    print("Total read time: ", tot_time)
