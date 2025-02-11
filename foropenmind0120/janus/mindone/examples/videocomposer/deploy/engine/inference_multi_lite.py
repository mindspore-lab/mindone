import logging
import os
from functools import partial
from typing import Any, Dict, List

import numpy as np
import tqdm
from mindspore_lite import Model

from ..config import Config, cfg
from ..modules import (
    CLIPTextProcessor,
    prepare_condition_models,
    prepare_dataloader,
    prepare_lite_model_kwargs,
    prepare_transforms,
    prepare_unet,
)
from ..schedulers import DiffusionSampler
from ..utils import (
    MSLiteModelBuilder,
    init_lite_infer,
    lite_predict,
    make_masked_images,
    swap_c_t_and_tile,
    visualize_with_model_kwargs,
)

__all__ = ["inference_multi_lite"]

_logger = logging.getLogger(__name__)


def inference_multi_lite(cfg_update: Dict[str, Any], **kwargs: Any) -> None:
    cfg.update(**kwargs)

    for k, v in cfg_update.items():
        cfg[k] = v

    cfg.save_origin_video = getattr(cfg, "save_origin_video", True)

    _inference_multi_lite(cfg)


def _inference_multi_lite(cfg: Config) -> None:
    input_video_name = os.path.basename(cfg.input_video).split(".")[0]
    init_lite_infer(cfg, video_name=input_video_name)

    lite_builder = MSLiteModelBuilder(
        device_target=cfg.device_target, device_id=cfg.device_id, lite_model_root=cfg.model_root
    )
    transforms_list = prepare_transforms(cfg)
    dataloader = prepare_dataloader(cfg, transforms_list)
    tokenizer = CLIPTextProcessor(cfg.clip_tokenizer)

    clip_encoder = lite_builder("clip_encoder")
    clip_encoder_visual = lite_builder("clip_encoder_visual")

    depth_extractor, canny_extractor, sketch_extractor = prepare_condition_models(lite_builder, cfg)

    # define tasks
    tasks = [
        ["y", "motion"],
        ["y", "sketch"],
        ["y", "depth"],
        ["y", "local_image"],
        ["image", "depth"],
        ["y", "masked"],
    ]

    models = [prepare_unet(lite_builder, task, sample_scheduler=cfg.sample_scheduler) for task in tasks]
    decoder = lite_builder("decoder")

    # global variables
    batch_size = cfg.batch_size
    frames_num = cfg.max_frames

    for step, batch in enumerate(dataloader.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if cfg.max_frames == 1 and cfg.use_image_dataset:
            ref_imgs, caps, _, misc_data, mask, mv_data = batch
            fps = np.array([cfg.feature_framerate] * batch_size, dtype=np.int64)
        else:
            ref_imgs, caps, _, misc_data, fps, mask, mv_data = batch

        caps = caps.tolist()

        for trial in tqdm.trange(cfg.n_iter, desc="trial"):
            if "motion" in cfg.video_compositions:
                mv_data_video = swap_c_t_and_tile(mv_data)
            else:
                mv_data_video = None

            # mask images
            if "mask" in cfg.video_compositions:
                masked_video = make_masked_images(misc_data, mask)
                masked_video = swap_c_t_and_tile(masked_video)
            else:
                masked_video = None

            # Single Image
            if "local_image" in cfg.video_compositions:
                image_local = np.tile(misc_data[:, :1], (1, frames_num, 1, 1, 1))
                image_local = swap_c_t_and_tile(image_local)
            else:
                image_local = None

            # preprocess for input text descripts
            text_tensor = tokenizer(caps)
            text_embs = lite_predict(clip_encoder, text_tensor)
            empty_text_tensor = tokenizer([""] * batch_size)
            empty_text_embs = lite_predict(clip_encoder, empty_text_tensor)
            if cfg.use_fps_condition:
                text_embs = np.concatenate([text_embs, np.zeros_like(text_embs)])
            else:
                text_embs = np.concatenate([text_embs, empty_text_embs])
            empty_text_embs = np.concatenate([empty_text_embs, empty_text_embs])

            # preprocess for input image
            if "image" in cfg.video_compositions:
                img_embs = lite_predict(clip_encoder_visual, ref_imgs)
                img_embs = np.expand_dims(img_embs, 1)
                img_embs = np.concatenate([img_embs, np.zeros_like(img_embs)])
            else:
                img_embs = None

            if "depthmap" in cfg.video_compositions:
                depth_data = lite_predict(depth_extractor, misc_data)
            else:
                depth_data = None

            if "canny" in cfg.video_compositions:
                canny_data = canny_extractor(misc_data)
            else:
                canny_data = None

            if "sketch" in cfg.video_compositions:
                sketch_data = lite_predict(sketch_extractor, misc_data)
            else:
                sketch_data = None

            if "single_sketch" in cfg.video_compositions:
                single_sketch_data = np.tile(sketch_data[:, :, :1], (1, 1, frames_num, 1, 1))
            else:
                single_sketch_data = None

            c = cfg.unet_in_dim
            h = cfg.resolution // 2 ** (len(cfg.unet_dim_mult) - 1)
            w = cfg.resolution // 2 ** (len(cfg.unet_dim_mult) - 1)
            if cfg.share_noise:
                noise = np.random.randn(batch_size, c, h, w).astype(np.float32)
                noise = np.tile(np.expand_dims(noise, 2), (1, 1, frames_num, 1, 1))
            else:
                noise = np.random.randn(batch_size, c, frames_num, h, w).astype(np.float32)

            full_model_kwargs = {
                "y": text_embs,
                "local_image": image_local,
                "image": img_embs,
                "depth": depth_data,
                "canny": canny_data,
                "sketch": sketch_data,
                "masked": masked_video,
                "motion": mv_data_video,
                "single_sketch": single_sketch_data,
                "fps": fps,
            }

            # Save generated videos
            infer = partial(
                _infer_with_partial_keys,
                full_model_kwargs=full_model_kwargs,
                decoder=decoder,
                noise=noise,
                misc=misc_data,
                empty_text_embs=empty_text_embs,
                caps=caps,
                step=step,
                trial=trial,
                cfg=cfg,
            )

            for task, model in zip(tasks, models):
                diffusion = DiffusionSampler(
                    model, scheduler_name=cfg.sample_scheduler, num_timesteps=cfg.num_timesteps
                )
                infer(task, diffusion=diffusion)

    _logger.info("Congratulations! The inference is completed!")
    _logger.info(f"The output is saved at `{cfg.log_dir}`.")


def _infer_with_partial_keys(
    partial_keys: List[str],
    full_model_kwargs: Dict[str, np.ndarray],
    diffusion: DiffusionSampler,
    decoder: Model,
    noise: np.ndarray,
    misc: np.ndarray,
    empty_text_embs: np.ndarray,
    caps: List[str],
    step: int,
    trial: int,
    cfg: Config,
) -> None:
    if "y" not in partial_keys:
        partial_keys.append("y")
        full_model_kwargs["y"] = empty_text_embs

    model_kwargs = prepare_lite_model_kwargs(
        partial_keys=partial_keys,
        full_model_kwargs=full_model_kwargs,
        use_fps_condition=cfg.use_fps_condition,
    )

    diffusion_output = diffusion(
        noise,
        model_kwargs=model_kwargs,
        guide_scale=cfg.guidance_scale,
        timesteps=cfg.sample_steps,
        eta=cfg.ddim_eta,
    )
    video_output = lite_predict(decoder, diffusion_output)

    fname = "-".join(partial_keys)
    visualize_with_model_kwargs(
        model_kwargs=model_kwargs,
        video_data=video_output,
        ori_video=misc,
        caps=caps,
        fname=f"{fname}_vid{step * cfg.n_iter + trial:04d}.gif",
        step=step,
        trial=trial,
        cfg=cfg,
    )
