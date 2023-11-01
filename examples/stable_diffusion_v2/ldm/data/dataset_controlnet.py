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

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from ldm.data.t2i_collate import data_column_controlnet, t2i_collate_controlnet
from ldm.data.dataset import ImageDataset, MetaLoader, build_dataloader_ft
from PIL import Image

from mindspore.dataset import GeneratorDataset
import logging
_logger = logging.getLogger(__name__)

SD_VERSION = os.getenv("SD_VERSION", default="2.0")
CONTROL_TYPE = {
    "canny": "canny",
}


# customize dataset for control net
def list_controlnet_image_files_captions_recursively(data_path):
    original_path = os.path.join(data_path, "original")
    control_path = os.path.join(data_path, "control")
    if not os.path.exists(original_path):
        raise ValueError(f"Data directory for original images {original_path} does not exist!")
    if not os.path.exists(control_path):
        raise ValueError(f"Data directory for control images {control_path} does not exist!")

    anno_dir = data_path
    anno_list = sorted(
        [os.path.join(anno_dir, f) for f in list(filter(lambda x: x.endswith(".csv"), os.listdir(anno_dir)))]
    )
    db_list = [pd.read_csv(f) for f in anno_list]
    all_images, all_captions, all_controls = [], [], []

    for db in db_list:
        all_images.extend(list(db["dir"]))
        all_captions.extend(list(db["text"]))
    assert len(all_images) == len(all_captions)
    all_originals = [os.path.join(original_path, f) for f in all_images]
    all_controls = [os.path.join(control_path, f) for f in all_images]

    return all_originals, all_captions, all_controls


class ControlImageDataset(ImageDataset):
    def __init__(
        self,
        batch_size,
        image_paths,
        captions,
        control_paths,
        tokenizer,
        control_type,
        image_size,
        image_filter_size,
        shuffle=True,
        random_crop=False,
        filter_small_size=False,
    ):
        super().__init__(
            batch_size,
            image_paths,
            captions,
            tokenizer,
            image_size,
            image_filter_size,
            shuffle=shuffle,
            random_crop=random_crop,
            filter_small_size=filter_small_size,
        )
        self.local_controls = control_paths
        self.control_type = control_type
    
    def preprocess_control(self, control_path):
        if self.control_type == "canny":
            control = Image.open(control_path)
            if control.mode == "L":
                control = np.array(control).astype(np.float32)/ 255 # to [0,1]
                control = np.expand_dims(control, axis=2)  # hw1
                control = np.concatenate([control, control, control], axis=2)  # hwc
            elif control.mode == "RGB":
                control = np.array(control).astype(np.float32)/ 255  # to [0,1]
            else:
                raise NotImplementedError(f"Process control image in {control.mode} is not implemented!")

        else:
            raise NotImplementedError(f"Control type {self.control_type} is not implemented!")
        return control

    def __getitem__(self, idx):
        # images preprocess
        img_path = self.local_images[idx]
        image_input = self.preprocess_image(img_path)

        # caption preprocess
        caption = self.local_captions[idx]
        caption_input = self.tokenize(caption)

        # control images
        control_path = self.local_controls[idx]
        control_input = self.preprocess_control(control_path)

        return (
            np.array(image_input, dtype=np.float32),
            np.array(caption_input, dtype=np.int32),
            np.array(control_input, dtype=np.float32),
        )

class ControlMetaLoader(MetaLoader):
    """For ControlNet"""

    def __init__(self, loaders, datalen, task_num=1):
        super().__init__(loaders, datalen, task_num)

    def get_batch(self, batch, task):
        """get_batch"""
        batch = defaultdict(lambda: None, batch)
        img_feat = batch.get("img_feat", None)
        txt_tokens = batch.get("txt_tokens", None)
        controls = batch.get("controls", None)
        output = (img_feat, txt_tokens, controls)

        return output



def load_data(
    data_path,
    batch_size,
    tokenizer,
    control_type,
    image_size=512,
    image_filter_size=256,
    device_num=1,
    random_crop=False,
    filter_small_size=True,
    rank_id=0,
    sample_num=-1,
):
    if not os.path.exists(data_path):
        raise ValueError(f"Data directory {data_path} does not exist!")
    all_images, all_captions, all_controls = list_controlnet_image_files_captions_recursively(data_path)

    _logger.debug(
        f"The first image path is {all_images[0]}, its control image path is {all_controls[0]}, and the caption is {all_captions[0]}"
    )
    _logger.info(f"Total number of training samples: {len(all_images)}")
    dataloaders = {}
    dataset = ControlImageDataset(
        batch_size,
        all_images,
        all_captions,
        all_controls,
        tokenizer,
        control_type,
        image_size,
        image_filter_size,
        random_crop=random_crop,
        filter_small_size=filter_small_size,
    )
    datalen = dataset.__len__

    loader = build_dataloader_ft(dataset, datalen, t2i_collate_controlnet, batch_size, device_num, rank_id=rank_id)
    dataloaders["ft_controlnet"] = loader
    if sample_num == -1:
        batchlen = datalen // (batch_size * device_num)
    else:
        batchlen = sample_num
    metaloader = ControlMetaLoader(dataloaders, datalen=batchlen, task_num=len(dataloaders.keys()))
    dataset = GeneratorDataset(metaloader, column_names=data_column_controlnet, shuffle=True)

    return dataset




