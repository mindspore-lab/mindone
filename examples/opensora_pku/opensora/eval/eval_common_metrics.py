"""Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could
measure the similarity of cross modalities. Please find more information from
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate
the mean average of cosine similarities.

See --help to see further details.

Code apapted from https://github.com/mseitzer/pytorch-fid and https://github.com/openai/CLIP.

Copyright 2023 The Hong Kong Polytechnic University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import os.path as osp
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from decord import VideoReader

sys.path.append(".")
from opensora.eval.cal_fvd import calculate_fvd
from opensora.eval.cal_lpips import calculate_lpips
from opensora.eval.cal_psnr import calculate_psnr

try:
    from opensora.eval.cal_flolpips import calculate_flolpips

    flolpips_isavailable = True
except Exception:
    flolpips_isavailable = False
from opensora.eval.cal_ssim import calculate_ssim
from opensora.models.ae.videobase.dataset_videobase import create_dataloader
from opensora.utils.dataset_utils import create_video_transforms
from tqdm import tqdm


class VideoDataset:
    def __init__(
        self,
        real_video_dir,
        generated_video_dir,
        num_frames,
        sample_rate=1,
        crop_size=None,
        resolution=128,
    ) -> None:
        super().__init__()
        self.real_video_files = self.combine_without_prefix(real_video_dir)
        self.generated_video_files = self.combine_without_prefix(generated_video_dir)
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.crop_size = crop_size
        self.short_size = resolution

        self.pixel_transforms = create_video_transforms(
            size=self.short_size,
            crop_size=crop_size,
            random_crop=False,
            disable_flip=True,
            num_frames=num_frames,
            transform_backend="al",
        )

    def __len__(self):
        return len(self.real_video_files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_video_file = self.real_video_files[index]
        generated_video_file = self.generated_video_files[index]
        print(real_video_file, generated_video_file)
        real_video_tensor = self._load_video(real_video_file)
        generated_video_tensor = self._load_video(generated_video_file)
        return {"real": real_video_tensor, "generated": generated_video_tensor}

    def _load_video(self, video_path):
        num_frames = self.num_frames
        sample_rate = self.sample_rate
        decord_vr = VideoReader(
            video_path,
        )
        total_frames = len(decord_vr)
        sample_frames_len = sample_rate * num_frames

        if total_frames >= sample_frames_len:
            s = 0
            e = s + sample_frames_len
            num_frames = num_frames
        else:
            s = 0
            e = total_frames
            num_frames = int(total_frames / sample_frames_len * num_frames)
            print(
                f"sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}",
                video_path,
                total_frames,
            )

        frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
        pixel_values = decord_vr.get_batch(frame_id_list).asnumpy()
        # video_data = video_data.transpose(0, 3, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        # NOTE:it's to ensure augment all frames in a video in the same way.
        # ref: https://albumentations.ai/docs/examples/example_multi_target/

        inputs = {"image": pixel_values[0]}
        for i in range(num_frames - 1):
            inputs[f"image{i}"] = pixel_values[i + 1]

        output = self.pixel_transforms(**inputs)

        pixel_values = np.stack(list(output.values()), axis=0)
        # (t h w c) -> (c t h w)
        pixel_values = np.transpose(pixel_values, (3, 0, 1, 2))
        pixel_values = pixel_values / 255.0
        return pixel_values

    def combine_without_prefix(self, folder_path, prefix="."):
        folder = []
        os.makedirs(folder_path, exist_ok=True)
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            if osp.isfile(osp.join(folder_path, name)):
                folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder


def calculate_common_metric(args, dataloader):
    score_list = []
    for batch_data in tqdm(dataloader):  # {'real': real_video_tensor, 'generated':generated_video_tensor }
        real_videos = batch_data["real"]
        generated_videos = batch_data["generated"]
        assert real_videos.shape[2] == generated_videos.shape[2]
        if args.metric == "fvd":
            tmp_list = list(calculate_fvd(real_videos, generated_videos, method=args.fvd_method)["value"].values())
        elif args.metric == "ssim":
            tmp_list = list(calculate_ssim(real_videos, generated_videos)["value"].values())
        elif args.metric == "psnr":
            tmp_list = list(calculate_psnr(real_videos, generated_videos)["value"].values())
        elif args.metric == "flolpips":
            if flolpips_isavailable:
                result = calculate_flolpips(
                    real_videos,
                    generated_videos,
                )
                tmp_list = list(result["value"].values())
            else:
                continue
        else:
            tmp_list = list(
                calculate_lpips(
                    real_videos,
                    generated_videos,
                )["value"].values()
            )
        score_list += tmp_list
    return np.mean(score_list)


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use")
    parser.add_argument("--real_video_dir", type=str, help=("the path of real videos`"))
    parser.add_argument("--generated_video_dir", type=str, help=("the path of generated videos`"))
    parser.add_argument("--device", type=str, default=None, help="Device to use. Like cuda, cuda:0 or cpu")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=("Number of processes to use for data loading. " "Defaults to `min(8, num_cpus)`"),
    )
    parser.add_argument("--sample_fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--metric", type=str, default="fvd", choices=["fvd", "psnr", "ssim", "lpips", "flolpips"])
    parser.add_argument("--fvd_method", type=str, default="styleganv", choices=["styleganv", "videogpt"])

    args = parser.parse_args()

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    dataset = VideoDataset(
        args.real_video_dir,
        args.generated_video_dir,
        num_frames=args.num_frames,
        sample_rate=args.sample_rate,
        crop_size=args.crop_size,
        resolution=args.resolution,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        ds_name="video",
        num_parallel_workers=num_workers,
        shuffle=False,
        drop_remainder=False,
    )
    metric_score = calculate_common_metric(args, dataloader)
    print("metric: ", args.metric, " ", metric_score)


if __name__ == "__main__":
    main()
