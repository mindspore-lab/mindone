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
import sys

import cv2
import numpy as np

sys.path.append("../stable_diffusion_v2")  # FIXME: loading modules from the stable_diffusion_v2 directory
from ldm.data.dataset import MetaLoader, build_dataloader_ft
from ldm.data.t2i_collate import data_column, t2i_collate

from mindspore.dataset import GeneratorDataset


def read_video_frames(video_path, sample_interval=None, image_size=None, sample_start_index=0):
    """
    Read frames from video path.
    If frame_rate is specified, read frames under the frame rate.
    If image_size is specified, resize video frames.
    """
    vidcap = cv2.VideoCapture(video_path)
    if sample_interval is not None:
        interval = sample_interval
    else:
        interval = 1

    video_frames = []
    frame_index = 0

    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret:
            frame_index += 1
            actual_frame_index = frame_index - sample_start_index
            if actual_frame_index % interval == 0:
                h, w, _ = frame.shape
                size = (w, h)
                if image_size is not None:
                    size = image_size
                frame = cv2.resize(frame, size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frames.append(frame)
        else:
            break

    vidcap.release()
    if len(video_frames) == 0:
        print("The sampled frames are empty. Please check whether the video file is empty or adjust the frame rate")
    return np.array(video_frames)


class TuneAVideoDataset:
    def __init__(
        self,
        video_path,
        prompt,
        tokenizer,
        image_size,
        num_frames: int = 24,
        sample_start_idx: int = 0,
        sample_interval: int = 2,
    ):
        self.video_path = video_path
        if not isinstance(image_size, (list, tuple)):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.num_frames = num_frames
        self.sample_start_idx = sample_start_idx
        self.sample_interval = sample_interval
        self.tokenizer = tokenizer

        self.video_frames = read_video_frames(
            self.video_path, sample_interval=sample_interval, image_size=image_size, sample_start_index=sample_start_idx
        )
        self.prompt = prompt
        self.prompt_ids = self.tokenize(self.prompt)

    @property
    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        # vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        # sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.num_frames]
        video = self.video_frames[: self.num_frames]
        video = (video / 127.5 - 1.0).astype(np.float32)
        prompt_ids = self.prompt_ids.astype(np.int64)
        return video, prompt_ids

    def tokenize(self, text):
        SOT_TEXT = self.tokenizer.sot_text  # "[CLS]"
        EOT_TEXT = self.tokenizer.eot_text  # "[SEP]"
        CONTEXT_LEN = 77  # TODO: get from self.tokenizer.context_len

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        result = np.zeros([CONTEXT_LEN]) + eot_token
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]
        result[: len(tokens)] = tokens

        return result


def load_data(
    video_path,
    prompt,
    tokenizer,
    batch_size,
    image_size=512,
    num_frames: int = 24,
    sample_start_idx: int = 0,
    sample_interval: int = 2,
    device_num=1,
    rank_id=0,
    sample_num=-1,
):
    if not os.path.exists(video_path):
        raise ValueError(f"Data directory {video_path} does not exist!")
    dataloaders = {}
    dataset = TuneAVideoDataset(
        video_path,
        prompt,
        tokenizer,
        image_size=image_size,
        num_frames=num_frames,
        sample_start_idx=sample_start_idx,
        sample_interval=sample_interval,
    )

    datalen = dataset.__len__

    loader = build_dataloader_ft(dataset, datalen, t2i_collate, batch_size, device_num, rank_id=rank_id)
    dataloaders["tuneavideo"] = loader
    if sample_num == -1:
        batchlen = datalen // (batch_size * device_num)
    else:
        batchlen = sample_num
    metaloader = MetaLoader(dataloaders, datalen=batchlen, task_num=len(dataloaders.keys()))

    dataset = GeneratorDataset(metaloader, column_names=data_column, shuffle=True)

    print("dataset size per shard:", dataset.get_dataset_size(), flush=True)
    return dataset
