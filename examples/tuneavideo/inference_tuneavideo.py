"""
Text to video generation: Tune-A-Video
"""
import argparse
import logging
import os
import sys
import time

import imageio
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import mindspore as ms

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workspace)
sys.path.append("../stable_diffusion_v2")  # FIXME: loading modules from the stable_diffusion_v2 directory
from data.dataset_tuneavideo import load_data
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.uni_pc import UniPCSampler
from ldm.modules.logger import set_logger
from ldm.modules.train.tools import set_random_seed
from ldm.util import instantiate_from_config, str2bool
from utils import model_utils

logger = logging.getLogger("text_to_image")

# naming: {sd_base_version}-{variation}
_version_cfg = {
    "2.1": ("sd_v2-1_base-7c8d09ce.ckpt", "v2-inference.yaml", 512),
    "2.1-v": ("sd_v2-1_768_v-061732d1.ckpt", "v2-vpred-inference.yaml", 768),
    "2.0": ("sd_v2_base-57526ee4.ckpt", "v2-inference.yaml", 512),
    "2.0-v": ("sd_v2_768_v-e12e3a9b.ckpt", "v2-vpred-inference.yaml", 768),
    "1.5": ("sd_v1.5-d0ab7146.ckpt", "v1-inference.yaml", 512),
    "1.5-wukong": ("wukong-huahua-ms.ckpt", "v1-inference-chinese.yaml", 512),
}
_URL_PREFIX = "https://download.mindspore.cn/toolkits/mindone/stable_diffusion"
_MIN_CKPT_SIZE = 4.0 * 1e9


def load_model_from_config(config, ckpt, **kwargs):
    model = instantiate_from_config(config.model)

    def _load_model(_model, ckpt_fp, verbose=True, filter=None):
        if os.path.exists(ckpt_fp):
            param_dict = ms.load_checkpoint(ckpt_fp)
            if param_dict:
                param_not_load, ckpt_not_load = model_utils.load_param_into_net_with_filter(
                    _model, param_dict, filter=filter
                )
                if verbose:
                    if len(param_not_load) > 0:
                        logger.info(
                            "Net params not loaded: {}".format([p for p in param_not_load if not p.startswith("adam")])
                        )
        else:
            logger.error(f"!!!Error!!!: {ckpt_fp} doesn't exist")
            raise FileNotFoundError(f"{ckpt_fp} doesn't exist")

    logger.info(f"Loading model from {ckpt}")
    _load_model(model, ckpt)

    model.set_train(False)
    for param in model.trainable_params():
        param.requires_grad = False

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        nargs="?",
        default="",
        help="the source (reference) video path.",
    )
    parser.add_argument("--num_frames", default=8, type=int, help="the number of sampled frames from the input video")
    parser.add_argument("--sample_start_idx", default=0, type=int, help="the sample start index of the frames")
    parser.add_argument(
        "--sample_interval",
        default=1,
        type=int,
        help="the sampling interval of frames. sample_interval=2 means to decrease the frame rate by 2.",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        nargs="?",
        default="2.1",
        help="Stable diffusion version. Options: '2.1', '2.1-v', '2.0', '2.0-v', '1.5', '1.5-wukong'",
    )
    parser.add_argument(
        "--prompt", type=str, nargs="?", default="", help="the prompt to render the video from the reference video"
    )
    parser.add_argument("--negative_prompt", type=str, nargs="?", default="", help="the negative prompt not to render")
    parser.add_argument("--output_path", type=str, nargs="?", default="output", help="dir to write results to")
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )

    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps. The recommended value is 50 for PLMS, DDIM and 20 for UniPC,DPM-Solver, DPM-Solver++",
    )
    parser.add_argument(
        "--inv_sampling_steps", type=int, default=50, help="The sampling steps to get the DDIM inversion latents."
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--use_inv_latent",
        default=True,
        type=str2bool,
        help="Use DDIM inversion latents",
    )
    parser.add_argument("--fps", type=int, default=8, help="the frame rate for the saved videos.")
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="number of iterations or trials. sample this often, ",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="use ddim sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action="store_true",
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--plms",
        action="store_true",
        help="use plms sampling",
    )
    parser.add_argument(
        "--uni_pc",
        action="store_true",
        help="use uni_pc sampling",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="unconditional guidance scale: eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
        "Simplified: `uc + scale * (uc - prompt)`",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model. If None, select by version",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    args = parser.parse_args()

    # check args
    if args.version:
        if args.version not in _version_cfg:
            raise ValueError(f"Unknown version: {args.version}. Supported SD versions are: {list(_version_cfg.keys())}")
    assert args.ckpt_path is not None, "ckpt_path must be specified!"

    assert args.config is not None, "config must be specified!"
    assert os.path.exists(args.video_path), f"the reference video {args.video_path} does not exsit!"

    if args.scale is None:
        args.scale = 9.0 if args.version.startswith("2.") else 7.5
    return args


def main(args):
    # set logger
    set_logger(
        name="",
        output_dir=args.output_path,
        rank=0,
        log_level=eval(args.log_level),
    )

    work_dir = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"WORK DIR:{work_dir}")
    os.makedirs(args.output_path, exist_ok=True)
    outpath = args.output_path
    batch_size = args.n_samples

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # set ms context
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(mode=args.ms_mode, device_target="Ascend", device_id=device_id, max_device_memory="30GB")

    set_random_seed(args.seed)

    # create model
    if not os.path.isabs(args.config):
        args.config = os.path.join(work_dir, args.config)
    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(
        config,
        ckpt=args.ckpt_path,
    )

    prediction_type = getattr(config.model, "prediction_type", "noise")
    logger.info(f"Prediction type: {prediction_type}")
    # create sampler
    if args.ddim:
        sampler = DDIMSampler(model)
        sname = "ddim"
    elif args.dpm_solver:
        sampler = DPMSolverSampler(model, "dpmsolver", prediction_type=prediction_type)
        sname = "dpm_solver"
    elif args.plms:
        sampler = PLMSSampler(model)
        sname = "plms"
    elif args.uni_pc:
        sampler = UniPCSampler(model)
        sname = "uni_pc"
    else:
        sampler = DPMSolverSampler(model, "dpmsolver++", prediction_type=prediction_type)
        sname = "dpm_solver_pp"
    if prediction_type == "v":
        assert sname in [
            "dpm_solver",
            "dpm_solver_pp",
        ], "Only dpm_solver and dpm_solver_pp support v-prediction currently."

    if args.use_inv_latent:
        inv_sampler = DDIMSampler(model)
        inv_sampler.make_schedule(args.inv_sampling_steps, verbose=False)
    # prepare prompt and reference video
    negative_prompt = args.negative_prompt
    assert negative_prompt is not None
    negative_data = [batch_size * [negative_prompt]]

    prompt = args.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]
    # read the source reference video
    tokenizer = model.cond_stage_model.tokenizer
    dataset = load_data(
        args.video_path,
        "",  # we don't use the reference video's prompt, so put an empty prompt here
        tokenizer,
        args.n_samples,
        image_size=(args.H, args.W),
        num_frames=args.num_frames,
        sample_start_idx=args.sample_start_idx,
        sample_interval=args.sample_interval,
    )
    for batch in dataset:
        frames, _ = batch
        break

    # log
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
            "Distributed mode: False",
            f"Number of input prompts: {len(data)}",
            f"Number of input negative prompts: {len(negative_data)}",
            f"Number of trials for each prompt: {args.n_iter}",
            f"Number of samples in each trial: {args.n_samples}",
            f"Model: StableDiffusion v-{args.version}",
            f"Precision: {model.model.diffusion_model.dtype}",
            f"Pretrained ckpt path: {args.ckpt_path}",
            f"Sampler: {sname}",
            f"Sampling steps: {args.sampling_steps}",
            f"Uncondition guidance scale: {args.scale}",
            f"Target image size (H, W): ({args.H}, {args.W})",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    # infer
    start_code = None
    if args.fixed_code:
        stdnormal = ms.ops.StandardNormal()
        start_code = stdnormal((args.n_samples, 4, args.num_frames, args.H // 8, args.W // 8))

    for i, prompts in enumerate(data):
        negative_prompts = negative_data[i]
        logger.info(
            f"[{i + 1}/{len(data)}] Generating videos with conditions:\nPrompt(s): {prompts[0]}\n"
            f"Negative prompt(s): {negative_prompts[0]}"
        )
        for n in range(args.n_iter):
            start_time = time.time()
            uc = None
            if args.scale != 1.0:
                if isinstance(negative_prompts, tuple):
                    negative_prompts = list(negative_prompts)
                tokenized_negative_prompts = model.tokenize(negative_prompts)
                uc = model.get_learned_conditioning(tokenized_negative_prompts)
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            tokenized_prompts = model.tokenize(prompts)
            c = model.get_learned_conditioning(tokenized_prompts)
            shape = [4, args.num_frames, args.H // 8, args.W // 8]

            if args.use_inv_latent:
                # get vae encode
                empty_c = model.get_learned_conditioning(model.tokenize([""] * batch_size))
                latents, _ = model.get_input(frames, empty_c)
                ddim_inv, _ = inv_sampler.encode(latents, empty_c, args.inv_sampling_steps)
                start_code = ddim_inv

            samples_ddim, _ = sampler.sample(
                S=args.sampling_steps,
                conditioning=c,
                batch_size=args.n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=args.scale,
                unconditional_conditioning=uc,
                eta=args.ddim_eta,
                x_T=start_code,
            )

            b, c, f, h, w = samples_ddim.shape
            samples_ddim = samples_ddim.transpose((0, 2, 1, 3, 4)).reshape((b * f, c, h, w))
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = ms.ops.clip_by_value((x_samples_ddim + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
            _, c, h, w = x_samples_ddim.shape
            x_samples_ddim = x_samples_ddim.reshape((b, f, c, h, w))  # (b, f, c, h, w)
            x_samples_ddim_numpy = x_samples_ddim.asnumpy()

            if not args.skip_save:
                for x_sample in x_samples_ddim_numpy:
                    x_sample = 255.0 * x_sample.transpose((0, 2, 3, 1))  # (f, h, w, c)
                    imgs = [x.astype(np.uint8) for x in x_sample]
                    try:
                        imageio.mimsave(os.path.join(sample_path, f"{base_count:05}.gif"), imgs, fps=args.fps)
                    except Exception:
                        duration = int(1000 / args.fps)
                        imageio.mimsave(os.path.join(sample_path, f"{base_count:05}.gif"), imgs, duration=duration)
                    # save images to subfolders
                    i_frame = 0
                    for img in imgs:
                        subfolder = os.path.join(sample_path, f"{base_count:05}")
                        os.makedirs(subfolder, exist_ok=True)
                        img = Image.fromarray(img)
                        img.save(os.path.join(subfolder, f"{i_frame:05}.png"))
                        i_frame += 1
                    base_count += 1

            end_time = time.time()
            logger.info(
                f"{batch_size * (n + 1)}/{batch_size * args.n_iter} images generated, "
                f"time cost for current trial: {end_time - start_time:.3f}s"
            )

    logger.info(f"Done! All generated videos are saved in: {outpath}/samples" f"\nEnjoy.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
