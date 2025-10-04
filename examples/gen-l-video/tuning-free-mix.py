import argparse
import copy
import logging
import os
import sys

import cv2
import numpy as np
from glv.pipelines.pipeline_tuning_free import TuningFreePipeline
from glv.util import ddim_inversion_long, save_videos_grid
from omegaconf import OmegaConf
from PIL import Image

import mindspore as ms
from mindspore import ops

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workspace + "/../stable_diffusion_v2")

from conditions.depth import DepthEstimator
from depth_to_image import load_model_from_config
from ldm.modules.train.tools import set_random_seed
from ldm.util import count_params, instantiate_from_config
from utils.download import download_checkpoint

_URL_PREFIX = "https://download.mindspore.cn/toolkits/mindone/stable_diffusion"

logger = logging.getLogger(__name__)


def prepare_video(cfg):
    cap = cv2.VideoCapture(cfg.video_path)
    pil_frames = []
    resized_frames = []

    while cap.isOpened():
        ret, cv2_frame = cap.read()

        if ret:
            converted_cv2_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(converted_cv2_frame))
            resized_frames.append(
                np.array(pil_frames[-1].resize((cfg.width, cfg.height), resample=Image.LANCZOS), dtype=np.float32)
            )
        else:
            break

    # sample frames
    sample_index = list(range(cfg.sample_start_idx, len(pil_frames), cfg.sample_frame_rate))[: cfg.n_sample_frames]

    pil_frames = [pil_frames[i] for i in sample_index]
    resized_frames = [resized_frames[i] for i in sample_index]

    resized_frames = np.stack(resized_frames, axis=0)

    resized_frames = resized_frames / 255.0
    resized_frames = resized_frames * 2 - 1

    resized_frames = ms.Tensor(resized_frames, dtype=ms.float32)

    return pil_frames, resized_frames


def main(args):
    # set ms context
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(mode=args.ms_mode, device_id=device_id)

    # set random seed
    set_random_seed(args.seed)

    # load configs
    sd_config = OmegaConf.load(f"{args.sd_config}")
    unet3d_config = OmegaConf.load(f"{args.unet3d_config}")
    scheduler_config = OmegaConf.load(f"{args.scheduler_config}")
    glv_config = OmegaConf.load(f"{args.glv_config}")

    train_data = glv_config.train_data
    validation_data = glv_config.validation_data

    # Handle the output folder creation
    output_dir = glv_config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
    OmegaConf.save(glv_config, os.path.join(output_dir, "glv.yaml"))

    # construct modules
    sd = load_model_from_config(sd_config, args.sd_ckpt_path)
    sd.model = None  # discard original unet2d of sd
    unet = load_model_from_config(unet3d_config, args.unet3d_ckpt_path)
    noise_scheduler = instantiate_from_config(scheduler_config)
    depth_estimator = DepthEstimator(estimator_ckpt_path=args.depth_ckpt_path, amp_level=glv_config.amp_level)

    num_params_sd, _ = count_params(sd)
    num_params_unet3d, _ = count_params(unet)
    num_params_depth_estimator, _ = count_params(depth_estimator.depth_estimator)

    if glv_config.amp_level != "O0":
        unet = ms.amp.auto_mixed_precision(unet, amp_level=glv_config.amp_level)
        custom_black_list = ms.amp.get_black_list()
        custom_black_list.append(ms.nn.GroupNorm)
        unet = ms.amp.custom_mixed_precision(unet, black_list=custom_black_list)

    print(
        f"Total Number of Parameters: "
        f"{(num_params_sd + num_params_unet3d + num_params_depth_estimator) * 1.e-6:.2f} M params.",
        flush=True,
    )

    # build validation pipeline
    validation_pipeline_depth = TuningFreePipeline(sd, unet, noise_scheduler, depth_estimator)

    ddim_inv_scheduler = instantiate_from_config(scheduler_config)
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # inference process
    pil_frames, pixel_values = prepare_video(train_data)
    video_length = pixel_values.shape[0]
    video_length = video_length - video_length % validation_data.video_length

    pil_frames = pil_frames[:video_length]
    pixel_values = pixel_values[:video_length]

    latents = []

    for i in range(0, video_length, validation_data.video_length):
        h = sd.first_stage_model.encoder(pixel_values[i : i + validation_data.video_length].permute(0, 3, 1, 2))
        latents.append(sd.get_first_stage_encoding(sd.first_stage_model.quant_conv(h).chunk(2, axis=1)[0]))

    latents = ops.cat(latents, axis=0)
    latents = latents.reshape(
        latents.shape[0] // video_length, video_length, latents.shape[1], latents.shape[2], latents.shape[3]
    ).permute(0, 2, 1, 3, 4)

    if glv_config.amp_level != "O0":
        latents = latents.to(ms.float32)

    samples = []
    ddim_inv_latent = None
    clip_length = validation_data.video_length

    if glv_config.run_isolated:
        if validation_data.use_inv_latent:
            # Convert videos to latent space
            ddim_inv_latent_lst = []

            for i in range(0, video_length - clip_length + 1, clip_length):
                ddim_inv_latent = ddim_inversion_long(
                    validation_pipeline_depth,
                    ddim_inv_scheduler,
                    video_latent=latents[:, :, i : i + clip_length],
                    num_inv_steps=validation_data.num_inv_steps,
                    prompt="",
                    window_size=clip_length,
                    stride=clip_length,
                    pixel_values=pil_frames[i : i + clip_length],
                )[-1]
                ddim_inv_latent_lst.append(ddim_inv_latent)

            ddim_inv_latent = ops.cat(ddim_inv_latent_lst, axis=2)
            inv_latents_path = os.path.join(output_dir, "inv_latents/ddim_latent-iso.npy")
            np.save(inv_latents_path, ddim_inv_latent.asnumpy())

        for prompt in validation_data.prompts:
            sample_lst = []
            assert len(prompt) == video_length // clip_length

            for i in range(0, video_length - clip_length + 1, clip_length):
                validation_isodata = copy.deepcopy(validation_data)
                validation_isodata.stride = clip_length
                sample = validation_pipeline_depth(
                    prompt[i // clip_length],
                    pil_frames[i : i + clip_length],
                    latents=ddim_inv_latent[:, :, i : i + clip_length],
                    window_size=clip_length,
                    **validation_data,
                ).videos
                sample_lst.append(sample)

            sample = ops.cat(sample_lst, axis=2)
            save_videos_grid(sample, f"{output_dir}/samples/sample-iso/{prompt}.gif", rescale=True)
            samples.append(sample)

        samples = ops.concat(samples)
        save_path = f"{output_dir}/samples/sample-iso.gif"
        save_videos_grid(samples, save_path, rescale=True)
        logging.info(f"Saved samples to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ms_mode",
        type=int,
        default=1,
        help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0).",
    )

    parser.add_argument(
        "--sd_config",
        type=str,
        default=None,
        help="Path to config for Stable Diffusion.",
    )

    parser.add_argument(
        "--unet3d_config",
        type=str,
        default=None,
        help="Path to the config for UNet3D.",
    )

    parser.add_argument(
        "--scheduler_config",
        type=str,
        default=None,
        help="Path to the config for scheduler.",
    )

    parser.add_argument(
        "--glv_config",
        type=str,
        default=None,
        help="Path to the config for Gen-L-Video.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed (for reproducible sampling).",
    )

    parser.add_argument(
        "--sd_ckpt_path",
        type=str,
        default=None,
        help="Path to the checkpoint of Stable Diffusion.",
    )

    parser.add_argument(
        "--unet3d_ckpt_path",
        type=str,
        default=None,
        help="Path to the checkpoint of UNet3D.",
    )

    parser.add_argument(
        "--depth_ckpt_path",
        type=str,
        default=None,
        help="Path to the checkpoint of depth estimator.",
    )

    args = parser.parse_args()

    # overwrite env var by parsed arg
    if args.sd_config is None:
        args.sd_config = "configs/tuning-free-mix/sd.yaml"

    if args.unet3d_config is None:
        args.unet3d_config = "configs/tuning-free-mix/unet3d.yaml"

    if args.scheduler_config is None:
        args.scheduler_config = "configs/tuning-free-mix/scheduler.yaml"

    if args.glv_config is None:
        args.glv_config = "configs/tuning-free-mix/glv.yaml"

    if args.sd_ckpt_path is None:
        ckpt_name = "sd_v2_depth-186e18a0.ckpt"
        args.sd_ckpt_path = "models/" + ckpt_name

        if not os.path.exists(args.sd_ckpt_path):
            print(f"Start downloading checkpoint {ckpt_name} ...")
            download_checkpoint(os.path.join(_URL_PREFIX, ckpt_name), "models/")

    if args.unet3d_ckpt_path is None:
        args.unet3d_ckpt_path = "models/diffusion_pytorch_model.ckpt"

    if args.depth_ckpt_path is None:
        args.depth_ckpt_path = "models/depth_estimator/midas_v3_dpt_large-c8fd1049.ckpt"

    main(args)
