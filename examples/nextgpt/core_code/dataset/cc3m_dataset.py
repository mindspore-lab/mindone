#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
import numpy as np
import json
from .base_dataset import BaseDataset
from tqdm import tqdm
import pandas as pd
from .utils import process_caption


class CC3MDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, embed_path: str, dataset_type: str):
        super(CC3MDataset, self).__init__(data_path, mm_root_path, embed_path, dataset_type)
        self.embed_path = embed_path

        print('Load CC3M dataset ...')
        self.mm_path_list, self.caption_list = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for row in tqdm(data, total=len(data)):
            image_id, one_caption = row["image_name"], row["caption"]
            self.mm_path_list.append(os.path.join(mm_root_path, image_id))
            self.caption_list.append(process_caption(one_caption))

        print(f'[!] collect {len(self.mm_path_list)} samples for training')
        self.dataset_type_list = [dataset_type for _ in range(len(self.caption_list))]

