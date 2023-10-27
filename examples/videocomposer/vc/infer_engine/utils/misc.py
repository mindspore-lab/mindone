import datetime
import json
import logging
import os
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

import mindspore as ms

from ...config import Config
from ...utils import save_video_multiple_conditions, setup_logger, setup_seed

__all__ = ["init_infer", "read_image_if_provided", "visualize_with_model_kwargs"]

_logger = logging.getLogger(__name__)


def init_infer(cfg: Config, video_name: Optional[str] = None) -> None:
    ms.set_context(mode=cfg.ms_mode, ascend_config={"precision_mode": "allow_fp32_to_fp16"})

    # logging
    log_dir = cfg.log_dir
    if video_name is None:
        ct = datetime.datetime.now().strftime("-%y%m%d%H%M")
        exp_name = os.path.basename(cfg.cfg_file).split(".")[0] + "-S%05d" % (cfg.seed) + ct
    else:
        exp_name = os.path.basename(cfg.cfg_file).split(".")[0] + f"-{video_name}" + "-S%05d" % (cfg.seed)
    log_dir = os.path.join(log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    cfg.log_dir = log_dir
    setup_logger(output_dir=cfg.log_dir)
    _logger.info(cfg)

    setup_seed(cfg.seed)


def read_image_if_provided(
    flag: bool, path: str, transform: Optional[Callable] = None, dtype: np.dtype = np.float32
) -> np.ndarray:
    "read image `path` if `flag` is True, else return None"
    if not flag:
        return None
    img = Image.open(path).convert("RGB")
    if transform is not None:
        img = transform(img)
    img = np.array(img, dtype=dtype)
    return img


def visualize_with_model_kwargs(
    model_kwargs: Dict[str, np.ndarray],
    video_data: np.ndarray,
    ori_video: np.ndarray,
    caps: List[str],
    fname: str,
    step: int,
    trial: int,
    cfg: Config,
) -> None:
    # remove the duplicated model_kwargs
    for key, conditions in model_kwargs.items():
        model_kwargs[key] = np.split(conditions, 2)[0]

    oss_key = os.path.join(cfg.log_dir, fname)

    # Save videos and text inputs.
    del model_kwargs[list(model_kwargs.keys())[0]]  # remove y
    save_video_multiple_conditions(
        oss_key,
        video_data,
        model_kwargs,
        ori_video,
        cfg.mean,
        cfg.std,
        save_origin_video=cfg.save_origin_video,
        save_frames=cfg.save_frames,
    )

    # add inputs info
    info_root = os.path.splitext(oss_key)[0]
    if not os.path.isdir(info_root):
        os.makedirs(info_root)
    info_json = os.path.join(info_root, "inputs.json")
    with open(info_json, "w") as f:
        content = {
            "file_name": fname,
            "prompt": caps,
            "batch_num": step,
            "trial_num": trial,
        }
        json.dump(content, f, indent=4)
