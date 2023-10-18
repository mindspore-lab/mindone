from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from mindspore_lite import Model

import mindspore.dataset as ds
from mindspore.dataset import transforms, vision

from ..config import Config
from ..data import CenterCrop, CenterCrop_Array, RandomResize, VideoDataset
from ..utils import MSLiteModelBuilder
from .extractor import CannyExtractor

__all__ = [
    "prepare_condition_models",
    "prepare_unet",
    "prepare_lite_model_kwargs",
    "prepare_dataloader",
    "prepare_transforms",
]


def prepare_condition_models(
    lite_builder: MSLiteModelBuilder, cfg: Config
) -> Tuple[Optional[Model], Optional[Model], Optional[Model]]:
    if not hasattr(cfg, "guidances"):
        cfg["guidances"] = ["depth", "canny", "sketch"]

    # [Conditions] Generators for various conditions
    if "depthmap" in cfg.video_compositions and "depth" in cfg.guidances:
        depth_extractor = lite_builder("depth_extractor_guidance")
    else:
        depth_extractor = None

    if "canny" in cfg.video_compositions and "canny" in cfg.guidances:
        canny_extractor = CannyExtractor()
    else:
        canny_extractor = None

    if "sketch" in cfg.video_compositions and ("single_sketch" in cfg.guidances or "sketch" in cfg.guidances):
        sketch_extractor = lite_builder("sketch_extractor_guidance")
    else:
        sketch_extractor = None

    return depth_extractor, canny_extractor, sketch_extractor


def prepare_unet(lite_builder: MSLiteModelBuilder, task: List[str], sample_scheduler: str = "DDIM") -> Model:
    if "y" not in task:
        task.append("y")

    task_model_names = f"{'-'.join(sorted(task))}_{sample_scheduler}_model"
    model = lite_builder(task_model_names)
    return model


def prepare_lite_model_kwargs(
    partial_keys: Iterable[str], full_model_kwargs: Dict[str, np.ndarray], use_fps_condition: bool
) -> Dict[str, np.ndarray]:
    allowed_keys = {
        "y",
        "depth",
        "canny",
        "masked",
        "sketch",
        "image",
        "motion",
        "local_image",
        "single_sketch",
    }

    for partial_key in partial_keys:
        assert partial_key in allowed_keys

    if use_fps_condition is True:
        partial_keys.append("fps")

    partial_model_kwargs = dict()
    for partial_key in partial_keys:
        if isinstance(full_model_kwargs[partial_key], np.ndarray):
            partial_model_kwargs[partial_key] = full_model_kwargs[partial_key]
        else:
            raise TypeError(f"Unsupported type `{type(full_model_kwargs[partial_key])}` for `{partial_key}`")

    return partial_model_kwargs


def prepare_dataloader(
    cfg: Config, transforms_list: Tuple[Callable, Callable, Callable, Callable]
) -> ds.GeneratorDataset:
    infer_transforms, misc_transforms, mv_transforms, vit_transforms = transforms_list
    dataset = VideoDataset(
        cfg=cfg,
        max_words=cfg.max_words,
        feature_framerate=cfg.feature_framerate,
        max_frames=cfg.max_frames,
        image_resolution=cfg.resolution,
        transforms=infer_transforms,
        mv_transforms=mv_transforms,
        misc_transforms=misc_transforms,
        vit_transforms=vit_transforms,
        vit_image_size=cfg.vit_image_size,
        misc_size=cfg.misc_size,
        mvs_visual=cfg.mvs_visual,
    )
    dataloader = ds.GeneratorDataset(
        source=dataset,
        column_names=["ref_frame", "cap_txt", "video_data", "misc_data", "feature_framerate", "mask", "mv_data"],
    )
    dataloader = dataloader.batch(cfg.batch_size)
    return dataloader


def prepare_transforms(cfg: Config) -> Tuple[Callable, Callable, Callable, Callable]:
    # [Transform] Transforms for different inputs
    infer_transforms = transforms.Compose(
        [
            CenterCrop(size=cfg.resolution),
            vision.ToTensor(),
            vision.Normalize(mean=cfg.mean, std=cfg.std, is_hwc=False),
        ]
    )
    misc_transforms = transforms.Compose(
        [
            RandomResize(size=cfg.misc_size),
            CenterCrop(cfg.misc_size),
            vision.ToTensor(),
        ]
    )
    mv_transforms = transforms.Compose(
        [
            vision.Resize(size=cfg.resolution),
            CenterCrop_Array(cfg.resolution),
        ]
    )
    vit_transforms = transforms.Compose(
        [
            CenterCrop(cfg.vit_image_size),
            vision.ToTensor(),
            vision.Normalize(mean=cfg.vit_mean, std=cfg.vit_std, is_hwc=False),
        ]
    )
    return infer_transforms, misc_transforms, mv_transforms, vit_transforms
