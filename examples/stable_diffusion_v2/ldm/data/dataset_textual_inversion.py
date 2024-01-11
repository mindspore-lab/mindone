# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import logging
import os
import random

import albumentations
import imagesize
import numpy as np
from ldm.data.commons import TEMPLATES_FACE, TEMPLATES_OBJECT, TEMPLATES_STYLE
from ldm.data.dataset import MetaLoader, build_dataloader_ft
from ldm.data.t2i_collate import data_column, t2i_collate
from PIL import Image
from PIL.ImageOps import exif_transpose

from mindspore.communication.management import get_local_rank, get_local_rank_size
from mindspore.dataset import GeneratorDataset

_logger = logging.getLogger(__name__)


def list_image_files(
    image_path,
    img_extensions=[".png", ".jpg", ".jpeg"],
):
    assert os.path.exists(image_path), f"The given data path {image_path} does not exist!"
    image_path_list = sorted(os.listdir(image_path))
    all_images = [
        os.path.join(image_path, f) for f in image_path_list if any([f.lower().endswith(e) for e in img_extensions])
    ]
    return all_images


def repeat_data(data_list, repeats):
    return data_list * repeats


def filter_small_image(all_images, image_filter_size, replace):
    filted_images = []
    filter_count = 0
    for image in all_images:
        try:
            w, h = imagesize.get(image)
        except Exception:
            filter_count += 1
            _logger.info(f"image file open failed or not exist(replace with others), path: {image}")
            continue
        if min(w, h) < image_filter_size:
            _logger.info(f"The size of image {image}: {w}x{h} < `image_filter_size` and excluded from training.")
            filter_count += 1
            continue
        else:
            filted_images.append(image)
    _logger.info(f"filter image count: {filter_count}")
    if replace:
        while filter_count > 0:
            filted_images.append(filted_images[filter_count])
            filter_count -= 1
    _logger.info("complete image list, size: " + str(len(filted_images)))
    return filted_images


def load_data(
    data_path,
    batch_size,
    tokenizer,
    train_data_repeats,
    learnable_property="object",
    templates=None,
    image_size=512,
    image_filter_size=256,
    random_crop=False,
    shuffle=True,
    filter_small_size=True,
    device_num=1,
    rank_id=0,
    replace=True,
    sample_num=-1,
    enable_modelarts=False,
    placeholder_token="*",
):
    if not os.path.exists(data_path):
        raise ValueError("Training data path directory does not exist!")
    train_images = list_image_files(data_path)
    if filter_small_size:
        train_images = filter_small_image(train_images, image_filter_size, replace)
    _logger.info(f"Total number of training samples: {len(train_images)}")

    train_images = repeat_data(train_images, train_data_repeats)
    _logger.info(
        f"The training data is repeated {train_data_repeats} times, and the total number is {len(train_images)}"
    )

    dataloaders = {}
    dataset = TextualInversionDataset(
        batch_size,
        train_images,
        tokenizer,
        image_size,
        learnable_property,
        shuffle=shuffle,
        random_crop=random_crop,
        placeholder_token=placeholder_token,
        templates=templates,
    )

    datalen = dataset.__len__
    if enable_modelarts:
        device_num = get_local_rank_size()
        rank_id = get_local_rank() % 8
    loader = build_dataloader_ft(dataset, datalen, t2i_collate, batch_size, device_num, rank_id=rank_id)
    dataloaders["ftT2I"] = loader
    if sample_num == -1:
        batchlen = datalen // (batch_size * device_num)
    else:
        batchlen = sample_num
    metaloader = MetaLoader(dataloaders, datalen=batchlen, task_num=len(dataloaders.keys()))

    dataset = GeneratorDataset(metaloader, column_names=data_column, shuffle=True)

    _logger.info("dataset size per shard: {}".format(dataset.get_dataset_size()))
    return dataset


class TextualInversionDataset:
    def __init__(
        self,
        batch_size,
        image_paths,
        tokenizer,
        image_size,
        learnable_property="object",
        shuffle=True,
        random_crop=False,
        filter_small_size=False,
        placeholder_token="*",
        templates=None,
    ):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.learnable_property = learnable_property
        self.local_images = image_paths
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.filter_small_size = filter_small_size
        self.placeholder_token = placeholder_token

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.image_size)
        if not self.random_crop:
            self.cropper = albumentations.CenterCrop(height=self.image_size, width=self.image_size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.cropper = albumentations.RandomCrop(height=self.image_size, width=self.image_size)
            self.preprocessor = albumentations.Compose(
                [self.rescaler, self.cropper, albumentations.HorizontalFlip(p=0.5)]
            )
            print("apply random crop and horizontal flip")

        # handle templates
        if templates is not None:
            assert (
                isinstance(templates, (list, tuple)) and len(templates) > 0
            ), "Expect to have non-empty templates as list or tuple."
            templates = list(templates)
            assert all(
                ["{}" in x for x in templates]
            ), "Expect to have templates list of strings such as 'a photo of {{}}'"
        else:
            if learnable_property.lower() == "object":
                templates = TEMPLATES_OBJECT
            elif learnable_property.lower() == "style":
                templates = TEMPLATES_STYLE
            elif learnable_property.lower() == "face":
                templates = TEMPLATES_FACE
            else:
                raise ValueError(
                    f"{learnable_property} learnable property is not supported! Only support ['object', 'style', 'face']"
                )
        self.templates = templates

    def __getitem__(self, idx):
        # images preprocess
        img_path = self.local_images[idx]
        image_input = self.preprocess_image(img_path)

        # caption generation
        placeholder_string = self.placeholder_token
        caption = random.choice(self.templates).format(placeholder_string)
        caption_input = self.tokenize(caption)

        return np.array(image_input, dtype=np.float32), np.array(caption_input, dtype=np.int32)

    # use CLIPTokenizer
    def tokenize(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.tokenizer.context_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
        )
        tokens = np.array(batch_encoding["input_ids"], np.int32)

        return tokens

    @property
    def __len__(self):
        return len(self.local_images)

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
        except Exception:
            print("image file open failed or not exist, path:", image_path, flush=True)
            image = np.zeros((512, 512, 3)).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image
