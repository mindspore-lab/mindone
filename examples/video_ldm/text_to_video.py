import time
from datetime import datetime
from pathlib import Path

import numpy as np
from jsonargparse import ArgumentParser
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.logger import set_logger
from ldm.modules.train.tools import set_random_seed
from omegaconf import OmegaConf
from PIL import Image

import mindspore as ms

from examples.stable_diffusion_v2.text_to_image import load_model_from_config


def main(args):
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # set logger
    logger = set_logger(
        name="Text-to-Video with VideoLDM",
        output_dir=args.output_path,
        rank=0,
    )

    # read prompts
    batch_size = args.n_samples
    prompt = batch_size * [args.prompt]
    negative_prompt = batch_size * [args.negative_prompt]

    sample_path = output_path / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sample_path.mkdir(exist_ok=True, parents=True)

    # set ms context
    ms.context.set_context(mode=args.ms_mode, device_target="Ascend")
    set_random_seed(args.seed)

    # create model
    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(
        config,
        ckpt=args.ckpt_path,
    )

    # FIXME: add AMP support

    prediction_type = getattr(config.model, "prediction_type", "noise")
    logger.info(f"Prediction type: {prediction_type}")
    # create sampler
    sampler = DDIMSampler(model)
    sname = "ddim"

    # log
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
            "Distributed mode: False",
            f"Prompt: {prompt}",
            f"Negative prompt: {negative_prompt}",
            f"Number of trials for each prompt: {args.n_iter}",
            f"Number of samples in each trial: {args.n_samples}",
            f"Model: StableDiffusion v-{args.version}",
            f"Precision: {model.model.diffusion_model.dtype}",
            f"Pretrained ckpt path: {args.ckpt_path}",
            f"Sampler: {sname}",
            f"Sampling steps: {args.sampling_steps}",
            f"Uncondition guidance scale: {args.scale}",
            f"Target video size (H, W): ({args.H}, {args.W})",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    # infer
    start_code = None
    if args.fixed_code:
        stdnormal = ms.ops.StandardNormal()
        start_code = stdnormal((args.n_samples, 4, args.H // 8, args.W // 8))

    for n in range(args.n_iter):
        start_time = time.perf_counter()
        uc = None
        if args.scale != 1.0:
            tokenized_negative_prompts = model.tokenize(negative_prompt)
            uc = model.get_learned_conditioning(tokenized_negative_prompts)

        tokenized_prompts = model.tokenize(prompt)
        c = model.get_learned_conditioning(tokenized_prompts)
        shape = [4, args.H // 8, args.W // 8]

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
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = np.clip((x_samples_ddim.asnumpy() + 1.0) / 2.0, 0.0, 1.0)

        frames = [
            Image.fromarray((255.0 * x_sample.transpose(1, 2, 0)).astype(np.uint8)) for x_sample in x_samples_ddim
        ]
        frames[0].save(
            sample_path / f"{n:05}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=1000 // 3,  # TODO: adjust FPS
            loop=0,
        )

        logger.info(
            f"{(n + 1)}/{args.n_iter} videos generated,"
            f" time cost for current trial: {time.perf_counter() - start_time:.3f}s"
        )

    logger.info(f"Done! All generated videos are saved in: {output_path}/samples\nEnjoy.")


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="2.0",
        choices=["1.5", "2.0", "2.1"],
        help="Stable diffusion version, 1.5, 2.0, or 2.1",
    )
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument(
        "--prompt", type=str, nargs="?", default="A cute wolf in winter forest", help="the prompt to render"
    )
    parser.add_argument("--negative_prompt", type=str, nargs="?", default="", help="the negative prompt not to render")
    parser.add_argument("--output_path", type=str, nargs="?", default="output", help="dir to write results to")
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps. The recommended value is  50 for PLMS, DDIM and 20 for UniPC,DPM-Solver, DPM-Solver++",
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
        "--n_iter",
        type=int,
        default=2,
        help="number of iterations or trials. sample this often, ",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="Video height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="Video width, in pixel space",
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
        default=None,
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
    args = parser.parse_args()

    if args.scale is None:
        args.scale = 9.0 if args.version.startswith("2.") else 7.5

    # core task
    main(args)
