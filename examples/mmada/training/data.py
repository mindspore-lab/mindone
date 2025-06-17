import itertools
import json
import math
import os
import random
import re
from functools import partial
from typing import List, Optional, Union

import pandas as pd
from PIL import Image

Image.warnings.simplefilter("error", Image.DecompressionBombWarning)

import webdataset as wds
import yaml
from braceexpand import braceexpand
from transformers import PreTrainedTokenizer
from webdataset.tariterators import base_plus_ext, tar_file_expander, url_opener, valid_sample

import mindspore.dataset.vision as transforms
from mindspore.dataset.vision import Inter

person_token = ["a person", "someone", "somebody"]


def replace_person_token(t):
    "Used for CC12M - handles all case variations of <person> tag"
    t = re.sub(r"<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t, flags=re.IGNORECASE)

    person_pattern = re.compile(r"<person>", re.IGNORECASE)
    while person_pattern.search(t):
        match = person_pattern.search(t)
        t = t[: match.start()] + f" {random.choice(person_token)} " + t[match.end() :]

    return t


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None, src=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if "fname" not in filesample.keys():
            print(f"fname not in filesample.keys(): {filesample}")
            print(f"src: {src}")
            continue
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()

        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw

    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)  # [{fname,data,__url__}, ...]  __url__ 字段标识当前读取的文件来自哪个 tar 包
    samples = group_by_keys_nothrow(files, handler=handler, src=src)
    return samples


def image_transform(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize(resolution, interpolation=Inter.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = (image - 0.5) / 0.5
    sample["images"] = image
    return sample


def image_transform_squash(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize((resolution, resolution), interpolation=Inter.BICUBIC)(image)
    image = transforms.ToTensor()(image)
    image = (image - 0.5) / 0.5
    sample["images"] = image
    return sample


def conditional_image_transform(sample, resolution=256):
    url = sample.get("__url__", "")
    special_datasets = ["ai2d", "clevr", "docvqa", "geo"]
    use_squash = False
    for keyword in special_datasets:
        if keyword in url:
            use_squash = True
            break
    if use_squash:
        return image_transform_squash(sample, resolution)
    else:
        return image_transform(sample, resolution)


def remove_prefix(caption):
    caption = (
        caption.replace("The image features ", "")
        .replace("The image presents ", "")
        .replace("The image you've sent is, ", "")
        .replace("In the center of the image, ", "")
        .replace("The image showcases ", "")
        .replace("The image is ", "")
        .replace("The image captures ", "")
        .replace("In the given image ", "")
        .replace("The image portrays ", "")
        .replace("In the image, ", "")
        .replace("In this image, we see ", "")
        .replace("The image depicts ", "")
        .replace("This is ", "")
        .replace("In this image, ", "")
        .replace("This image captures ", "")
    )

    return caption


def filter_long_samples(sample):
    return sample.get("input_ids") is not None


class Text2ImageDataset:
    def __init__(
        self,
        train_shards_path_or_url: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        num_train_examples: int,
        per_device_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        resolution: int = 256,
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        external_caption_path: Optional[str] = "",
        external_journeydb_caption_path: Optional[str] = "",
        external_laion12m_caption_path: Optional[str] = "",
        external_cc12m_caption_path: Optional[str] = "",
        external_text_to_image_2M_512_caption_path: Optional[str] = "",
        external_ai2d_caption_path: Optional[str] = "",
        external_clevr_caption_path: Optional[str] = "",
        external_docvqa_caption_path: Optional[str] = "",
        external_geo_caption_path: Optional[str] = "",
        is_captioning: bool = False,
        add_caption_prompt: bool = False,
        long_caption: bool = True,
        shuffle: bool = True,
    ):
        if f"{train_shards_path_or_url}.yaml" in os.listdir("./configs"):
            with open(f"./configs/{train_shards_path_or_url}.yaml") as f:
                train_shards_path_or_url = yaml.safe_load(f)
        self.long_caption = long_caption
        self.external_caption_path = external_caption_path
        self.external_journeydb_caption_path = external_journeydb_caption_path
        self.external_laion12m_caption_path = external_laion12m_caption_path
        self.external_cc12m_caption_path = external_cc12m_caption_path
        self.external_text_to_image_2M_512_caption_path = external_text_to_image_2M_512_caption_path
        self.is_captioning = is_captioning
        self.add_caption_prompt = add_caption_prompt
        if self.add_caption_prompt:
            with open("./training/questions.json") as f:
                self.caption_prompt = json.load(f)
                # self.caption_prompt = ['USER: \n' + prompt + ' ASSISTANT:' for prompt in self.caption_prompt]
                self.caption_prompt = [
                    "<|start_header_id|>user<|end_header_id|>\n"
                    + prompt
                    + "<eot_id><|start_header_id|>assistant<|end_header_id|>\n"
                    for prompt in self.caption_prompt
                ]
        else:
            self.caption_prompt = None

        if external_journeydb_caption_path != "":
            with open(external_journeydb_caption_path) as file:
                self.journeydb_caption = json.load(file)
        else:
            self.journeydb_caption = None

        if external_ai2d_caption_path != "":
            self.ai2d_caption = pd.read_csv(external_ai2d_caption_path)
        if external_clevr_caption_path != "":
            self.clevr_caption = pd.read_csv(external_clevr_caption_path)
        if external_docvqa_caption_path != "":
            self.docvqa_caption = pd.read_csv(external_docvqa_caption_path)
        if external_geo_caption_path != "":
            self.geo_caption = pd.read_csv(external_geo_caption_path)

        def tokenize(text):
            if tokenizer is not None:
                text = replace_person_token(text)

                encoding = tokenizer(
                    text, truncation=True, max_length=2 * max_seq_length, padding=False, return_tensors="pt"
                )
                full_input_ids = encoding.input_ids[0]

                if len(full_input_ids) > max_seq_length:
                    return None
                else:
                    return text
            else:
                return text

        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

        if external_caption_path != "":
            processing_pipeline = [
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.map(self.load_external_caption, handler=wds.ignore_and_continue),
                wds.rename(
                    images="jpg;png;jpeg;webp",
                    input_ids="text;txt;caption",
                    handler=wds.warn_and_continue,
                ),
                wds.map(partial(conditional_image_transform, resolution=resolution), handler=wds.warn_and_continue),
                wds.map(filter_keys(set(["images", "input_ids"]))),
                wds.map_dict(
                    input_ids=tokenize,
                    handler=wds.warn_and_continue,
                ),
                wds.select(filter_long_samples),
            ]
        else:
            processing_pipeline = [
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.rename(
                    images="jpg;png;jpeg;webp",
                    input_ids="text;txt;caption",
                    handler=wds.warn_and_continue,
                ),
                wds.map(partial(conditional_image_transform, resolution=resolution), handler=wds.warn_and_continue),
                wds.map(filter_keys(set(["images", "input_ids"]))),
                wds.map_dict(
                    input_ids=tokenize,
                    handler=wds.warn_and_continue,
                ),
                wds.select(filter_long_samples),
            ]

        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            # wds.batched(per_device_batch_size, partial=False, collation_fn=default_collate),
            wds.batched(per_device_batch_size, partial=False),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    def load_external_caption(self, sample):
        if "SA1B" in sample["__key__"] or "sa" in sample["__key__"]:
            captionf = f"{self.external_caption_path}/{sample['__key__'].split('/')[-1]}.txt"
            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.readlines()[0].replace("\n", "")
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample["txt"] = prompt + captions
                else:
                    sample["txt"] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample["txt"] = captions.split(".")[0]
                else:
                    sample["txt"] = captions

                sample["txt"] = remove_prefix(sample["txt"])

            return sample

        elif "laion" in sample["__url__"]:
            url_part = sample["__url__"].split("/")[-1].split(".")[0]
            key = sample["__key__"].split("/")[-1]
            captionf = os.path.join(self.external_laion12m_caption_path, url_part, f"{key}.caption")

            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.read().strip()
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample["txt"] = prompt + captions
                else:
                    sample["txt"] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample["txt"] = captions.split(".")[0]
                else:
                    sample["txt"] = captions

                sample["txt"] = remove_prefix(sample["txt"])

            return sample

        elif "cc12m" in sample["__url__"]:
            url_part = sample["__url__"].split("/")[-1].split(".")[0]
            key = sample["__key__"].split("/")[-1]
            captionf = os.path.join(self.external_cc12m_caption_path, url_part, f"{key}.caption")

            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.read().strip()
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample["txt"] = prompt + captions
                else:
                    sample["txt"] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample["txt"] = captions.split(".")[0]
                else:
                    sample["txt"] = captions
                sample["txt"] = remove_prefix(sample["txt"])

            return sample

        elif "text-to-image-2M" in sample["__url__"]:
            if "json" in sample and "prompt" in sample["json"]:
                captions = sample["json"]["prompt"]
            else:
                print(f"sample has no json or prompt: {sample}")
                captions = ""

            sample["txt"] = captions

            return sample

        elif "ai2d" in sample["__url__"]:
            key = sample["__key__"].split("/")[-1]
            df_row = self.ai2d_caption[self.ai2d_caption["image"].astype(str) == key + ".png"]
            if len(df_row) == 0:
                print(f"No captions available for key {sample['__key__']}")
                return sample
            elif len(df_row) > 1:
                # print(f"Multiple captions available for key {sample['__key__']}")
                df_row = df_row.sample(1)
            question = df_row["question"].values[0]
            solution = df_row["solution"].values[0]
            caption = (
                "<|start_header_id|>user<|end_header_id|>\n"
                "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed \
                    within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
                f"{question}\n"
                "<eot_id><|start_header_id|>assistant<|end_header_id|>\n"
                f"{solution}"
            )
            sample["txt"] = caption
            return sample

        elif "clevr" in sample["__url__"]:
            key = sample["__key__"].split("/")[-1]
            df_row = self.clevr_caption[self.clevr_caption["image"].astype(str) == key + ".jpg"]
            if len(df_row) == 0:
                print(f"No captions available for key {sample['__key__']}")
                return sample
            elif len(df_row) > 1:
                # print(f"Multiple captions available for key {sample['__key__']}")
                df_row = df_row.sample(1)
            question = df_row["question"].values[0]
            solution = df_row["solution"].values[0]
            caption = (
                "<|start_header_id|>user<|end_header_id|>\n"
                "You should first think about the reasoning process in the mind and then provide the user with the answer. \
                    The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
                f"{question}\n"
                "<eot_id><|start_header_id|>assistant<|end_header_id|>\n"
                f"{solution}"
            )
            sample["txt"] = caption
            return sample

        elif "docvqa" in sample["__url__"]:
            key = sample["__key__"].split("/")[-1]
            df_row = self.docvqa_caption[self.docvqa_caption["image"].astype(str) == key + ".png"]
            if len(df_row) == 0:
                print(f"No captions available for key {sample['__key__']}")
                return sample
            elif len(df_row) > 1:
                # print(f"Multiple captions available for key {sample['__key__']}")
                df_row = df_row.sample(1)
            question = df_row["question"].values[0]
            solution = df_row["solution"].values[0]
            caption = (
                "<|start_header_id|>user<|end_header_id|>\n"
                "You should first think about the reasoning process in the mind and then provide the user with the answer. \
                    The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
                f"{question}\n"
                "<eot_id><|start_header_id|>assistant<|end_header_id|>\n"
                f"{solution}"
            )
            sample["txt"] = caption
            return sample

        elif "geo" in sample["__url__"]:
            key = sample["__key__"].split("/")[-1]
            df_row = self.geo_caption[self.geo_caption["image"].astype(str) == key + ".jpg"]
            if len(df_row) == 0:
                print(f"No captions available for key {sample['__key__']}")
                return sample
            elif len(df_row) > 1:
                # print(f"Multiple captions available for key {sample['__key__']}")
                df_row = df_row.sample(1)
            question = df_row["question"].values[0]
            solution = df_row["solution"].values[0]
            caption = (
                "<|start_header_id|>user<|end_header_id|>\n"
                "You should first think about the reasoning process in the mind and then provide the user with the answer. \
                    The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
                f"{question}\n"
                "<eot_id><|start_header_id|>assistant<|end_header_id|>\n"
                f"{solution}"
            )
            sample["txt"] = caption
            return sample

        elif self.journeydb_caption is not None and sample["__key__"] in self.journeydb_caption:
            captions_list = self.journeydb_caption[sample["__key__"]]
            if len(captions_list) == 0:
                print(f"No captions available for key {sample['__key__']}")
                return sample
            sample["txt"] = random.sample(captions_list, 1)[0]
            return sample

        else:
            print(f"none exist sample: {sample}")
            return sample

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader


if __name__ == "__main__":
    pass
