import glob
import os
from pathlib import Path

import csv
import numpy as np
from PIL import Image

import mindspore as ms
import mindspore.dataset.transforms as transforms
import mindspore.ops as ops
from mindspore.dataset.vision import CenterCrop, Normalize, Resize, ToTensor

from mindone.visualize.videos import save_videos


def str2bool(b):
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True


def _get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list


def _load_prompts(prompt_file):
    f = open(prompt_file, "r")
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        line = line.strip()
        if len(line) != 0:
            prompt_list.append(line)
        f.close()
    return prompt_list


def load_data_prompts(prompt_csv, data_dir, img_col, text_col, video_size=(256, 256), video_frames=16, interp=False):
    transform = transforms.Compose(
        [
            Resize(min(video_size)),
            CenterCrop(video_size),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), is_hwc=False),
        ]
    )

    with open(prompt_csv, "r") as f:
        data = list(csv.DictReader(f))

    file_list = [line[img_col] for line in data]
    prompt_list = [line[text_col] for line in data]

    data_list = []
    filename_list = []
    n_samples = len(prompt_list)
    for idx in range(n_samples):
        if interp:
            raise NotImplementedError
        else:
            img_path = (Path(data_dir) / file_list[idx]).with_suffix(".png")  # TODO: support other img format
            image = Image.open(img_path).convert("RGB")
            image_tensor = ms.Tensor(transform(image)[0]).unsqueeze(1)  # [c,1,h,w]
            frame_tensor = ops.repeat_interleave(image_tensor, repeats=video_frames, axis=1)
            _, filename = os.path.split(file_list[idx])

        data_list.append(frame_tensor)
        filename_list.append(filename)

    return filename_list, data_list, prompt_list


def _transform_before_save(video):
    video = ops.transpose(video, (0, 2, 3, 4, 1))
    video = video.asnumpy()  # check the dtype
    video = np.clip(video, -1, 1)
    video = (video + 1.0) / 2.0
    return video


def save_results_seperate(prompt, samples, filename, fakedir, fps=10, loop=False):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    # save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        path = os.path.join(savedirs[idx], f'{filename.split(".")[0]}_sample{idx}.mp4')
        video_transform = _transform_before_save(video)
        save_videos(video_transform, path)
