import os
import random
import logging
import csv

import numpy as np
from decord import VideoReader
import mindspore as ms
from mindspore.dataset import transforms, vision
from mindspore import Tensor

_logger = logging.getLogger(__name__)

class WebVid:
    def __init__(self,
                 csv_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 **kwargs,
                 ):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        with open(self.csv_path, "r") as csv_file:
            self.metadata = list(csv.DictReader(csv_file))
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    vision.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    vision.Resize(min(self.resolution)),
                    vision.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
                self.spatial_transform = vision.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride

        max_attempt = 100  # TODO: can be optimized
        for _ in range(max_attempt):
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = os.path.join(self.data_dir, sample["video"])
            caption = sample['caption']

            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path)
                else:
                    video_reader = VideoReader(video_path, width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices).asnumpy()
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        frames = np.transpose(frames, (3, 0, 1, 2)).astype(np.float32)  # [t,h,w,c] -> [c,t,h,w]

        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        return frames, caption, video_path, fps_clip, frame_stride
    
    def __len__(self):
        return len(self.metadata)


def create_dataloader(config, device_num=1, rank_id=0):
    dataset = WebVid(**config)
    _logger.info(f"Total number of samples: {len(dataset)}")

    # Larger value leads to more memory consumption. Default: 16
    # prefetch_size = config.get("prefetch_size", 16)
    # ms.dataset.config.set_prefetch_size(prefetch_size)

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=config["column_names"],
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=config["shuffle"],
        num_parallel_workers=config["num_parallel_workers"],
        max_rowsize=config["max_rowsize"],  # video data require larger rowsize
    )

    dl = dataloader.batch(
        config["batch_size"],
        drop_remainder=True,
    )

    return dl
