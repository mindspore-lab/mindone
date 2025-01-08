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

import math
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(".")

from hyvideo.eval.cal_lpips import calculate_lpips
from hyvideo.eval.cal_psnr import calculate_psnr
from hyvideo.utils.dataset_utils import VideoPairDataset, create_dataloader

flolpips_isavailable = False
calculate_flolpips = None
from hyvideo.eval.cal_ssim import calculate_ssim
from tqdm import tqdm


def calculate_common_metric(args, dataloader, dataset_size):
    score_list = []
    index = 0
    for batch_data in tqdm(
        dataloader, total=dataset_size
    ):  # {'real': real_video_tensor, 'generated':generated_video_tensor }
        real_videos = batch_data["real"]
        generated_videos = batch_data["generated"]
        assert real_videos.shape[2] == generated_videos.shape[2]
        if args.metric == "fvd":
            if index == 0:
                print("calculate fvd...")
            raise ValueError
            # tmp_list = list(calculate_fvd(real_videos, generated_videos, method=args.fvd_method)["value"].values())
        elif args.metric == "ssim":
            if index == 0:
                print("calculate ssim...")
            tmp_list = list(calculate_ssim(real_videos, generated_videos)["value"].values())
        elif args.metric == "psnr":
            if index == 0:
                print("calculate psnr...")
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
            if index == 0:
                print("calculate_lpips...")
            tmp_list = list(
                calculate_lpips(
                    real_videos,
                    generated_videos,
                )["value"].values()
            )
        index += 1
        score_list += tmp_list
    return np.mean(score_list)


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use")
    parser.add_argument("--real_video_dir", type=str, help=("the path of real videos`"))
    parser.add_argument("--real_data_file_path", type=str, default=None, help=("the path of real videos csv file`"))
    parser.add_argument("--generated_video_dir", type=str, help=("the path of generated videos`"))
    parser.add_argument("--device", type=str, default=None, help="Device to use. Like GPU or Ascend")
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

    dataset = VideoPairDataset(
        args.real_video_dir,
        args.generated_video_dir,
        num_frames=args.num_frames,
        real_data_file_path=args.real_data_file_path,
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
    dataset_size = math.ceil(len(dataset) / float(args.batch_size))
    dataloader = dataloader.create_dict_iterator(1, output_numpy=True)
    metric_score = calculate_common_metric(args, dataloader, dataset_size)
    print("metric: ", args.metric, " ", metric_score)


if __name__ == "__main__":
    main()
