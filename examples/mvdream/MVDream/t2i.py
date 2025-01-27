import argparse
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

sys.path.append(
    os.path.realpath(os.path.join(os.path.dirname(__file__), "../../sv3d"))
)  # to include sv3d.sgm, so that instantiate_from_config works
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../..")))  # to include mindone

import logging
import time
from datetime import datetime

from mvdream.camera_utils import get_camera
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.ldm.util import instantiate_from_config
from mvdream.model_zoo import build_model

import mindspore as ms
from mindspore import mint

from mindone.utils.seed import set_random_seed


def log_file(output_dir=".", log_name="time_mea.out", log_level=logging.INFO):
    logger = logging.getLogger("")
    logger.setLevel(log_level)
    fmt = "%(asctime)s %(levelname)s: %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(os.path.join(output_dir, log_name))
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(file_handler)
    return logger


def t2i(
    model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0.0, camera=None, num_frames=1
):
    if type(prompt) != list:
        prompt = [prompt]
    c = model.get_learned_conditioning(prompt)
    c_ = {"context": c.tile((batch_size, 1, 1))}
    uc_ = {"context": uc.tile((batch_size, 1, 1))}
    if camera is not None:
        c_["camera"] = uc_["camera"] = camera
        c_["num_frames"] = uc_["num_frames"] = num_frames

    shape = [4, image_size // 8, image_size // 8]
    samples_ddim, _ = sampler.sample(
        S=step,
        conditioning=c_,
        batch_size=batch_size,
        shape=shape,
        verbose=False,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc_,
        eta=ddim_eta,
        x_T=None,
    )
    x_sample = model.decode_first_stage(samples_ddim)
    x_sample = mint.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).asnumpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="1", type=str, help="MindSpore execution mode: Graph mode[0] or Pynative mode[1]"
    )
    parser.add_argument(
        "--device_id", default="6", type=str, help="MindSpore execution mode: Graph mode[0] or Pynative mode[1]"
    )
    parser.add_argument(
        "--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="mvdream/configs/sd-v2-base.yaml",
        help="load model from local config (override model_name)",
    )
    parser.add_argument("--ckpt_path", type=str, default="sd-v2.1-base-4view.ckpt", help="path to local checkpoint")
    parser.add_argument("--text", type=str, default="an astronaut riding a horse")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=4, help="num of frames (views) to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--measure_time", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    set_random_seed(args.seed)
    dtype = ms.float32
    batch_size = max(4, args.num_frames)
    logger = log_file(log_name="t2i_" + args.text.replace(r" ", "_") + ".log")

    ms.set_context(mode=int(args.mode), device_target="Ascend")

    logger.info("load t2i model ... ")
    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    else:
        assert args.ckpt_path is not None, "ckpt_path must be specified!"
        config = OmegaConf.load(args.config_path)
        model = instantiate_from_config(config.model)
        m, u = ms.load_param_into_net(model, ms.load_checkpoint(args.ckpt_path), strict_load=False)
        if len(m) > 0 and args.verbose:
            logger.info("missing keys in params:")
            logger.info(m)
        if len(u) > 0 and args.verbose:
            logger.info("unexpected keys from ckpt:")
            logger.info(u)

    shape = [4, args.size // 8, args.size // 8]
    eps_shape = (batch_size, *shape)
    img_eps = np.random.randn(*eps_shape).astype(np.float32)
    # img_eps = np.load('th_eps_forthesame_noise.npy').astype(np.float32)  # for fixed noise, th vs. ms consistency
    sampler = DDIMSampler(model, img_eps=img_eps)
    uc = model.get_learned_conditioning([""])
    logger.info("load t2i model done . ")

    # pre-compute camera matrices
    if args.use_camera:
        camera = get_camera(
            args.num_frames,
            elevation=args.camera_elev,
            azimuth_start=args.camera_azim,
            azimuth_span=args.camera_azim_span,
        )
        camera = camera.tile((batch_size // args.num_frames, 1))
    else:
        camera = None

    t = args.text + args.suffix
    images = []
    batches_time = []
    if args.measure_time:
        nrun = 4
    else:
        nrun = 1
    for _ in range(nrun):
        start_time = time.time()
        img = t2i(
            model,
            args.size,
            t,
            uc,
            sampler,
            step=50,
            scale=10,
            batch_size=batch_size,
            ddim_eta=0.0,
            camera=camera,
            num_frames=args.num_frames,
        )
        img = np.concatenate(img, 1)
        images.append(img)
        if args.measure_time:
            batch_time = time.time() - start_time
            batches_time.append(batch_time)
            logger.info(f"Batch time cost: {batch_time:.3f}s.")

    if len(batches_time) > 1:
        del batches_time[0]
        mean_time = sum(batches_time) / len(batches_time)
        logger.info(f"Mean Batch time: {mean_time:.3f}s.")
        logger.info(f"Mean Batch speed: {(4 / mean_time):.3f}f/s.")

    if not args.debug:
        images = np.concatenate(images, 0)
        tstr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"sample_{tstr}_{args.text.replace(r' ', '_')}.png"
        Image.fromarray(images).save(save_path)
        logger.info("Sample saved at " + save_path)
