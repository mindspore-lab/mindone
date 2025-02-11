import logging
import time
from typing import Any, Dict

import numpy as np
import tqdm

from mindspore import Tensor

from ..config import Config, cfg
from .modules import (
    prepare_clip_encoders,
    prepare_condition_models,
    prepare_dataloader,
    prepare_decoder_unet,
    prepare_model_kwargs,
    prepare_model_visual_kwargs,
    prepare_transforms,
)
from .schedulers import DiffusionSampler
from .utils import (
    init_infer,
    make_masked_images,
    read_image_if_provided,
    swap_c_t_and_tile,
    visualize_with_model_kwargs,
)

__all__ = ["inference_single"]

_logger = logging.getLogger(__name__)


def inference_single(cfg_update: Dict[str, Any], **kwargs: Any) -> None:
    cfg.update(**kwargs)

    for k, v in cfg_update.items():
        cfg[k] = v

    cfg.read_image = getattr(cfg, "read_image", False)
    cfg.read_sketch = getattr(cfg, "read_sketch", False)
    cfg.read_style = getattr(cfg, "read_style", False)
    cfg.save_origin_video = getattr(cfg, "save_origin_video", True)

    _inference_single(cfg)


def _inference_single(cfg: Config) -> None:
    init_infer(cfg)

    transforms_list = prepare_transforms(cfg)
    misc_transforms = transforms_list[1]
    dataloader = prepare_dataloader(cfg, transforms_list)
    clip_encoder, clip_encoder_visual = prepare_clip_encoders(cfg)

    frame_in = read_image_if_provided(cfg.read_image, cfg.image_path, misc_transforms)
    frame_sketch = read_image_if_provided(cfg.read_sketch, cfg.sketch_path, misc_transforms)
    frame_style = read_image_if_provided(cfg.read_style, cfg.style_image, None)

    depth_extractor, canny_extractor, sketch_extractor = prepare_condition_models(cfg)

    decoder, model = prepare_decoder_unet(cfg)
    # diffusion
    diffusion = DiffusionSampler(model, scheduler_name=cfg.sample_scheduler, num_timesteps=cfg.num_timesteps)

    # global variables
    batch_size = cfg.batch_size
    frames_num = cfg.max_frames
    time_cost = np.zeros((dataloader.get_dataset_size(), cfg.n_iter))
    for step, batch in enumerate(dataloader.create_tuple_iterator(num_epochs=1)):
        if cfg.max_frames == 1 and cfg.use_image_dataset:
            ref_imgs, caps, _, misc_data, mask, mv_data = batch
            fps = np.array([cfg.feature_framerate] * batch_size, dtype=np.int64)
        else:
            ref_imgs, caps, _, misc_data, fps, mask, mv_data = batch

        caps = caps.asnumpy().tolist()

        for trial in tqdm.trange(cfg.n_iter, desc="trial"):
            start = time.time()
            if "motion" in cfg.video_compositions and "motion" in cfg.guidances:
                mv_data_video = swap_c_t_and_tile(mv_data)
            else:
                mv_data_video = None

            # mask images
            if "mask" in cfg.video_compositions and "masked" in cfg.guidances:  # TODO: fix the name
                masked_video = make_masked_images(misc_data, mask)
                masked_video = swap_c_t_and_tile(masked_video)
            else:
                masked_video = None

            # Single Image
            if "local_image" in cfg.video_compositions and "local_image" in cfg.guidances:
                if cfg.read_image:
                    image_local = np.tile(frame_in[:, None, ...], (batch_size, frames_num, 1, 1, 1))
                else:
                    image_local = np.tile(misc_data.asnumpy()[:, :1], (1, frames_num, 1, 1, 1))
                image_local = swap_c_t_and_tile(Tensor(image_local))
            else:
                image_local = None

            # preprocess for input text descripts
            if "text" in cfg.video_compositions and "y" in cfg.guidances:
                text_tensor = clip_encoder.preprocess(caps)
                text_embs = clip_encoder(text_tensor).asnumpy()  # [N, 77, 1024]
                if cfg.use_fps_condition:
                    text_embs = np.concatenate([text_embs, np.zeros_like(text_embs)])
                else:
                    empty_text_tensor = clip_encoder.preprocess([""] * batch_size)
                    empty_text_embs = clip_encoder(empty_text_tensor).asnumpy()
                    text_embs = np.concatenate([text_embs, empty_text_embs])
            else:
                empty_text_tensor = clip_encoder.preprocess([""] * batch_size)
                empty_text_embs = clip_encoder(empty_text_tensor).asnumpy()
                text_embs = np.concatenate([empty_text_embs, empty_text_embs])

            # preprocess for input image
            if "image" in cfg.video_compositions and "image" in cfg.guidances:
                if cfg.read_style:
                    frame_tensor = clip_encoder_visual.preprocess(frame_style)
                    img_embs = clip_encoder_visual(frame_tensor).asnumpy()
                    img_embs = np.expand_dims(img_embs, 0)
                else:
                    img_embs = clip_encoder_visual(ref_imgs).asnumpy()
                    img_embs = np.expand_dims(img_embs, 1)  # [N, 1, 1024]
                img_embs = np.concatenate([img_embs, np.zeros_like(img_embs)])
            else:
                img_embs = None

            # preprocess for misc dat
            if "depthmap" in cfg.video_compositions and "depth" in cfg.guidances:
                depth_data = depth_extractor(misc_data)
            else:
                depth_data = None

            if "canny" in cfg.video_compositions and "canny" in cfg.guidances:
                canny_data = canny_extractor(misc_data)
            else:
                canny_data = None

            if "sketch" in cfg.video_compositions and ("sketch" in cfg.guidances or "single_sketch" in cfg.guidances):
                if cfg.read_sketch:
                    sketch_data_input = np.tile(frame_sketch[:, None, ...], (batch_size, frames_num, 1, 1, 1))
                    sketch_data_input = Tensor(sketch_data_input)
                else:
                    sketch_data_input = misc_data
                sketch_data = sketch_extractor(sketch_data_input)
            else:
                sketch_data = None

            if "single_sketch" in cfg.video_compositions and "single_sketch" in cfg.guidances:
                single_sketch_data = np.tile(sketch_data.asnumpy()[:, :, :1], (1, 1, frames_num, 1, 1))
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
            noise = Tensor(noise)

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
            # --------------------------------------
            partial_keys = cfg.guidances
            model_kwargs = prepare_model_kwargs(
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
            video_output = decoder(diffusion_output)

            end = time.time()
            time_cost[step, trial] = end - start

            visualize_with_model_kwargs(
                model_kwargs=prepare_model_visual_kwargs(model_kwargs),
                video_data=video_output.asnumpy(),
                ori_video=misc_data.asnumpy(),
                caps=caps,
                fname=f"vid{step * cfg.n_iter + trial:04d}.gif",
                step=step,
                trial=trial,
                cfg=cfg,
            )

    _logger.info("Congratulations! The inference is completed!")
    _logger.info(f"The output is saved at `{cfg.log_dir}`.")
    _logger.info(
        f"Time cost: {time_cost[:, 0].mean():.3f}s (avg of the first trial); {time_cost[:, 1:].mean():.3f}s (avg of the remaining trials)"
    )
