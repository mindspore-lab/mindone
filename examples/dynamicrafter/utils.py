import glob
import os

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


def load_data_prompts(data_dir, video_size=(256, 256), video_frames=16, interp=False):
    transform = transforms.Compose(
        [
            Resize(min(video_size)),
            CenterCrop(video_size),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), is_hwc=False),
        ]
    )
    # load prompts
    prompt_file = _get_filelist(data_dir, ["txt"])
    assert len(prompt_file) > 0, "Error: found NO prompt file!"
    # default prompt
    default_idx = 0
    default_idx = min(default_idx, len(prompt_file) - 1)
    if len(prompt_file) > 1:
        print(f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
    # only use the first one (sorted by name) if multiple exist

    # load video
    file_list = _get_filelist(data_dir, ["jpg", "png", "jpeg", "JPEG", "PNG"])
    data_list = []
    filename_list = []
    prompt_list = _load_prompts(prompt_file[default_idx])
    n_samples = len(prompt_list)
    for idx in range(n_samples):
        if interp:
            image1 = Image.open(file_list[2 * idx]).convert("RGB")
            image_tensor1 = ms.Tensor(transform(image1)[0]).unsqueeze(1)  # [c,1,h,w]
            image2 = Image.open(file_list[2 * idx + 1]).convert("RGB")
            image_tensor2 = ms.Tensor(transform(image2)[0]).unsqueeze(1)  # [c,1,h,w]
            frame_tensor1 = ops.repeat_interleave(image_tensor1, repeats=video_frames // 2, axis=1)
            frame_tensor2 = ops.repeat_interleave(image_tensor2, repeats=video_frames // 2, axis=1)
            frame_tensor = ops.cat([frame_tensor1, frame_tensor2], axis=1)
            _, filename = os.path.split(file_list[idx * 2])
        else:
            image = Image.open(file_list[idx]).convert("RGB")
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
