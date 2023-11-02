import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore import dataset as ds
from mindspore.dataset import transforms, vision

from ...config import Config
from ...data import CenterCrop, CenterCrop_Array, RandomResize, VideoDataset
from ...models import AutoencoderKL, FrozenOpenCLIPEmbedder, FrozenOpenCLIPVisualEmbedder, UNetSD_temporal
from ...utils import get_abspath_of_weights
from .decoder import Decoder
from .extractor import CannyExtractor, ConditionExtractor, DepthExtractor, SketchExtractor

__all__ = [
    "prepare_clip_encoders",
    "prepare_condition_models",
    "prepare_dataloader",
    "prepare_decoder_unet",
    "prepare_model_kwargs",
    "prepare_model_visual_kwargs",
    "prepare_transforms",
]


_logger = logging.getLogger(__name__)


def prepare_clip_encoders(cfg: Config) -> Tuple[FrozenOpenCLIPEmbedder, FrozenOpenCLIPVisualEmbedder]:
    clip_encoder = FrozenOpenCLIPEmbedder(
        layer="penultimate",
        pretrained_ckpt_path=get_abspath_of_weights(cfg.clip_checkpoint),
        tokenizer_path=get_abspath_of_weights(cfg.clip_tokenizer),
        use_fp16=cfg.use_fp16,
    )
    clip_encoder_visual = FrozenOpenCLIPVisualEmbedder(
        layer="penultimate", pretrained_ckpt_path=get_abspath_of_weights(cfg.clip_checkpoint), use_fp16=cfg.use_fp16
    )
    return clip_encoder, clip_encoder_visual


def prepare_condition_models(
    cfg: Config,
) -> Tuple[Optional[ConditionExtractor], Optional[CannyExtractor], Optional[ConditionExtractor]]:
    if not hasattr(cfg, "guidances"):
        cfg["guidances"] = ["depth", "canny", "sketch"]

    # [Conditions] Generators for various conditions
    if "depthmap" in cfg.video_compositions and "depth" in cfg.guidances:
        depth_extractor = DepthExtractor(
            cfg.midas_checkpoint, depth_std=cfg.depth_std, depth_clamp=cfg.depth_clamp, use_fp16=cfg.use_fp16
        )
        depth_extractor = ConditionExtractor(depth_extractor, chunk_size=cfg.chunk_size)
    else:
        depth_extractor = None

    if "canny" in cfg.video_compositions and "canny" in cfg.guidances:
        canny_extractor = CannyExtractor()
    else:
        canny_extractor = None

    if "sketch" in cfg.video_compositions and ("single_sketch" in cfg.guidances or "sketch" in cfg.guidances):
        sketch_extractor = SketchExtractor(
            cfg.pidinet_checkpoint,
            cfg.sketch_simplification_checkpoint,
            sketch_mean=cfg.sketch_mean,
            sketch_std=cfg.sketch_std,
            use_fp16=cfg.use_fp16,
        )
        sketch_extractor = ConditionExtractor(sketch_extractor, chunk_size=cfg.chunk_size)
    else:
        sketch_extractor = None

    return depth_extractor, canny_extractor, sketch_extractor


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


def prepare_decoder_unet(cfg: Config, version: str = "2.1") -> Tuple[Decoder, UNetSD_temporal]:
    # [Model] autoencoder & unet
    autoencoder = AutoencoderKL(
        cfg.ddconfig,
        4,
        ckpt_path=get_abspath_of_weights(cfg.sd_checkpoint),
        use_fp16=cfg.use_fp16,
        version=version,
    )
    decoder = Decoder(autoencoder)

    if hasattr(cfg, "network_name") and cfg.network_name == "UNetSD_temporal":
        model = UNetSD_temporal(
            cfg=cfg,
            in_dim=cfg.unet_in_dim,
            concat_dim=cfg.unet_concat_dim,
            dim=cfg.unet_dim,
            context_dim=cfg.unet_context_dim,
            out_dim=cfg.unet_out_dim,
            dim_mult=cfg.unet_dim_mult,
            num_heads=cfg.unet_num_heads,
            head_dim=cfg.unet_head_dim,
            num_res_blocks=cfg.unet_res_blocks,
            attn_scales=cfg.unet_attn_scales,
            dropout=cfg.unet_dropout,
            temporal_attention=cfg.temporal_attention,
            temporal_attn_times=cfg.temporal_attn_times,
            use_checkpoint=cfg.use_checkpoint,
            use_fps_condition=cfg.use_fps_condition,
            use_sim_mask=cfg.use_sim_mask,
            video_compositions=cfg.video_compositions,
            misc_dropout=cfg.misc_dropout,
            p_all_zero=cfg.p_all_zero,
            p_all_keep=cfg.p_all_zero,
            use_fp16=cfg.use_fp16,
            use_adaptive_pool=False,
        )
    else:
        raise NotImplementedError(f"The model {cfg.network_name} not implement")

    # load checkpoint
    if cfg.resume and cfg.resume_checkpoint:
        model.load_state_dict(cfg.resume_checkpoint)
        _logger.info(f"Successfully load unet from {cfg.resume_checkpoint}")
    else:
        raise ValueError(f"The checkpoint file {cfg.resume_checkpoint} is wrong ")
    _logger.info(f"Created a model with {int(sum(p.size for p in model.get_parameters()) / (1024 ** 2))}M parameters")
    return decoder, model


def prepare_model_kwargs(
    partial_keys: List[str], full_model_kwargs: Dict[str, Union[np.ndarray, Tensor]], use_fps_condition: bool
) -> Dict[str, Tensor]:
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

    if "y" not in partial_keys:
        partial_keys.append("y")

    if use_fps_condition is True:
        partial_keys.append("fps")

    partial_model_kwargs = dict()
    for partial_key in partial_keys:
        if isinstance(full_model_kwargs[partial_key], ms.Tensor):
            partial_model_kwargs[partial_key] = full_model_kwargs[partial_key]
        elif isinstance(full_model_kwargs[partial_key], np.ndarray):
            partial_model_kwargs[partial_key] = Tensor(full_model_kwargs[partial_key])
        else:
            raise TypeError(f"Unsupported type `{type(full_model_kwargs[partial_key])}` for `{partial_key}`")

    return partial_model_kwargs


def prepare_model_visual_kwargs(model_kwargs: Dict[str, Tensor]) -> Dict[str, np.ndarray]:
    for k, v in model_kwargs.items():
        model_kwargs[k] = v.asnumpy()
    return model_kwargs


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
