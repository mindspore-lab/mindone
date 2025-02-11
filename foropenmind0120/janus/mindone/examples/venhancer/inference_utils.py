import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import Tensor, mint, ops

logger = logging.getLogger(__name__)


def tensor2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = Tensor(mean).reshape(1, -1, 1, 1, 1)
    std = Tensor(std).reshape(1, -1, 1, 1, 1)
    video = video.mul(std).add(mean)
    video = video.clamp(0, 1)
    video = video * 255.0
    # b c f h w -> b f h w c
    images = ops.transpose(video, (0, 2, 3, 4, 1))[0]
    return images


def preprocess(input_frames):
    out_frame_list = []
    for pointer in range(len(input_frames)):
        frame = input_frames[pointer]
        frame = frame[:, :, ::-1]
        frame = Image.fromarray(frame.astype("uint8")).convert("RGB")
        frame = ms.dataset.vision.ToTensor()(frame)
        out_frame_list.append(Tensor(frame))
    out_frames = ops.stack(out_frame_list, axis=0)
    out_frames.clamp(0, 1)
    mean = Tensor([0.5, 0.5, 0.5], dtype=out_frames.dtype).view(-1)
    std = Tensor([0.5, 0.5, 0.5], dtype=out_frames.dtype).view(-1)
    out_frames = mint.div(mint.sub(out_frames, mean.view(1, -1, 1, 1)), std.view(1, -1, 1, 1))
    return out_frames


def adjust_resolution(h, w, up_scale):
    if h * w * up_scale * up_scale < 720 * 1280 * 1.5:
        up_s = np.sqrt(720 * 1280 * 1.5 / (h * w))
        target_h = int(up_s * h // 2 * 2)
        target_w = int(up_s * w // 2 * 2)
    elif h * w * up_scale * up_scale > 1152 * 2048:
        up_s = np.sqrt(1152 * 2048 / (h * w))
        target_h = int(up_s * h // 2 * 2)
        target_w = int(up_s * w // 2 * 2)
    else:
        target_h = int(up_scale * h // 2 * 2)
        target_w = int(up_scale * w // 2 * 2)
    return (target_h, target_w)


def make_mask_cond(in_f_num, interp_f_num):
    mask_cond = []
    interp_cond = [-1 for _ in range(interp_f_num)]
    for i in range(in_f_num):
        mask_cond.append(i)
        if i != in_f_num - 1:
            mask_cond += interp_cond
    return mask_cond


def load_prompt_list(file_path):
    files = []
    with open(file_path, "r") as fin:
        for line in fin:
            path = line.strip()
            if path:
                files.append(path)
    return files


def load_video(vid_path):
    capture = cv2.VideoCapture(vid_path)
    _fps = capture.get(cv2.CAP_PROP_FPS)
    _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    pointer = 0
    frame_list = []
    stride = 1
    while len(frame_list) < _total_frame_num:
        ret, frame = capture.read()
        pointer += 1
        if (not ret) or (frame is None):
            break
        if pointer >= _total_frame_num + 1:
            break
        if pointer % stride == 0:
            frame_list.append(frame)
    capture.release()
    return frame_list, _fps


def save_video(video, save_dir, file_name, fps=16.0):
    output_path = os.path.join(save_dir, file_name)
    images = [(img.asnumpy()).astype("uint8") for img in video]
    temp_dir = tempfile.mkdtemp()
    for fid, frame in enumerate(images):
        tpth = os.path.join(temp_dir, "%06d.png" % (fid + 1))
        cv2.imwrite(tpth, frame[:, :, ::-1])
    tmp_path = os.path.join(save_dir, "tmp.mp4")
    cmd = f"ffmpeg -y -f image2 -framerate {fps} -i {temp_dir}/%06d.png \
      -crf 17 -pix_fmt yuv420p {tmp_path}"
    status, output = subprocess.getstatusoutput(cmd)
    if status != 0:
        logger.error(f"Save Video Error with {output}")
    os.system(f"rm -rf {temp_dir}")
    os.rename(tmp_path, output_path)
