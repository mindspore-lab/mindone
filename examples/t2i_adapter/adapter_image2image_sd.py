import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from adapters import CombinedAdapter, get_adapter
from omegaconf import OmegaConf
from PIL import Image
from t2i_utils import read_images

import mindspore as ms

sys.path.append("../stable_diffusion_v2/")
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.logger import set_logger
from ldm.modules.train.tools import set_random_seed
from ldm.util import str2bool
from text_to_image import load_model_from_config


def main(args):
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # read prompts
    batch_size = args.n_samples
    prompt = args.prompt
    negative_prompt = args.negative_prompt

    sample_path = output_path / "_".join(args.adapter_condition) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sample_path.mkdir(exist_ok=True, parents=True)

    # set ms context
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(mode=args.ms_mode, device_target="Ascend", device_id=device_id)
    if args.ms_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": args.jit_level})
    set_random_seed(args.seed)

    # create model
    if not Path(args.config).is_absolute():
        args.config = Path(__file__).parent / args.config
    config = OmegaConf.load(args.config)
    model = load_model_from_config(
        config,
        ckpt=args.ckpt_path,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_only_ckpt=args.lora_ckpt_path,
    )

    if len(args.cond_weight) == 1:  # if condition weights are not specified per adapter
        args.cond_weight *= len(args.adapter_condition)
    cond_weights = args.cond_weight

    assert (
        len(args.adapter_condition) == len(args.adapter_ckpt_path) == len(args.condition_image) == len(cond_weights)
    ), (
        f"Number of adapters and conditions should match, got {args.adapter_condition} adapters,"
        f" {args.adapter_ckpt_path} checkpoints, {args.condition_image} condition images, and {cond_weights} weights."
    )

    adapters = [
        get_adapter("sd", a_cond, ckpt, use_fp16=False)
        for a_cond, ckpt in zip(args.adapter_condition, args.adapter_ckpt_path)
    ]
    adapters = CombinedAdapter(adapters, cond_weights, output_fp16=model.dtype == ms.float16)

    cond_paths = args.condition_image
    assert all([os.path.isfile(cond) for cond in cond_paths]), "Paths to condition images must be files."

    # create sampler
    if args.ddim:
        sampler = DDIMSampler(model)
        sname = "ddim"
    elif args.plms:
        sampler = PLMSSampler(model)
        sname = "plms"
    else:
        raise ValueError("No sampler was specified. Supported samplers: ddim and plms.")

    # log
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
            "Distributed mode: False",
            f"Prompt: {prompt}",
            f"Negative prompt: {negative_prompt}",
            f"Conditions: {args.adapter_condition}",
            f"Condition images: {cond_paths}",
            f"Condition weights: {cond_weights}",
            f"Number of trials for each prompt: {args.n_iter}",
            f"Number of samples in each trial: {args.n_samples}",
            f"Model: StableDiffusion v-{args.version}",
            f"Precision: {model.model.diffusion_model.dtype}",
            f"Pretrained ckpt path: {args.ckpt_path}",
            f"Lora ckpt path: {args.lora_ckpt_path if args.use_lora else None}",
            f"Sampler: {sname}",
            f"Sampling steps: {args.sampling_steps}",
            f"Uncondition guidance scale: {args.scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    # infer
    start_code = None
    if args.fixed_code:
        stdnormal = ms.ops.StandardNormal()
        start_code = stdnormal((args.n_samples, 4, args.H // 8, args.W // 8))

    flags = []
    for condition in args.adapter_condition:
        if condition == "sketch":
            flags.append(0)
        else:
            flags.append(-1)
    conds, img_shape = read_images(cond_paths, min(args.H, args.W), flags=flags)
    args.H, args.W = img_shape
    adapter_features, context = adapters(conds)

    base_count = 0
    for n in range(args.n_iter):
        start_time = time.perf_counter()
        uc = None
        if args.scale != 1.0:
            tokenized_negative_prompts = model.tokenize([negative_prompt] * batch_size)
            uc = model.get_learned_conditioning(tokenized_negative_prompts)
        tokenized_prompts = model.tokenize([prompt] * batch_size)
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
            features_adapter=tuple(adapter_features) if isinstance(adapter_features, list) else adapter_features,
            append_to_context=context,
            cond_tau=args.cond_tau,
            style_cond_tau=args.style_cond_tau,
            eta=args.ddim_eta,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = np.clip((x_samples_ddim.asnumpy() + 1.0) / 2.0, 0.0, 1.0)

        for x_sample in x_samples_ddim:
            x_sample = 255.0 * x_sample.transpose(1, 2, 0)
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(sample_path / f"{base_count:05}.png")
            base_count += 1

        logger.info(
            f"{batch_size * (n + 1)}/{batch_size * args.n_iter} images generated, "
            f"time cost for current trial: {time.perf_counter() - start_time:.3f}s"
        )

    logger.info(f"Done! All generated images are saved in: {output_path}/samples\nEnjoy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Stable Diffusion
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports ['O0', 'O1', 'O2']."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument("--prompt", type=str, nargs="?", default="best quality", help="added prompt")
    parser.add_argument("--negative_prompt", type=str, nargs="?", default="", help="the negative prompt not to render")
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="2.0",
        choices=["1.5", "2.0", "2.1"],
        help="Stable diffusion version, 1.5, 2.0, or 2.1",
    )
    parser.add_argument("--output_path", type=str, nargs="?", default="output", help="dir to write results to")
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=50,
        help="Number of ddim sampling steps",
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
        default=2,
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
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="unconditional guidance scale: eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
        "Simplified: `uc + scale * (uc - prompt)`",
    )
    parser.add_argument(
        "--use_lora",
        default=False,
        type=str2bool,
        help="whether the checkpoint used for inference is finetuned from LoRA",
    )
    parser.add_argument(
        "--lora_rank",
        default=None,
        type=int,
        help="LoRA rank. If None, lora checkpoint should contain the value for lora rank in its append_dict.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--lora_ckpt_path",
        type=str,
        default=None,
        help="path to lora only checkpoint. Set it if use_lora is not None",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    # T2I-Adapter
    parser.add_argument(
        "--adapter_condition",
        type=str,
        nargs="+",
        help="Additional condition(s) for spatial (visual) guidance of SD with support of adapter."
        "Allows passing multiple conditions for Combined Adapters.",
    )
    parser.add_argument(
        "--adapter_ckpt_path",
        type=str,
        nargs="+",
        help="Path(s) to the adapter checkpoint(s).",
    )
    parser.add_argument(
        "--condition_image",
        type=str,
        nargs="+",
        help="Path(s) to a condition image(s).",
    )
    parser.add_argument(
        "--cond_tau",
        type=float,
        default=1.0,
        help="Timestamp parameter that determines until which step the adapter is applied. "
        "Similar as Prompt-to-Prompt tau.",
    )
    parser.add_argument(
        "--style_cond_tau",
        type=float,
        default=1.0,
        help="Timestamp parameter that determines until which step the adapter is applied. "
        "Similar as Prompt-to-Prompt tau",
    )
    parser.add_argument(
        "--cond_weight",
        type=float,
        nargs="+",
        default=[1.0],
        help="The adapter features are multiplied by the `cond_weight`. The larger the `cond_weight`, the more aligned "
        "the generated image and condition will be, but the generated quality may be reduced.",
    )

    args = parser.parse_args()

    # overwrite env var by parsed arg
    os.environ["SD_VERSION"] = args.version
    if args.version == "1.5":
        args.config = "../stable_diffusion_v2/configs/v1-inference.yaml"
        ckpt_path = "models/sd_v1.5-d0ab7146.ckpt"
    elif args.version == "2.0":
        args.config = "../stable_diffusion_v2/configs/v2-inference.yaml"
        ckpt_path = "models/sd_v2_base-57526ee4.ckpt"
    elif args.version == "2.1":
        args.config = "../stable_diffusion_v2/configs/v2-inference.yaml"
        ckpt_path = "models/sd_v2-1_base-7c8d09ce.ckpt"
    else:
        raise ValueError(f"Unsupported SD version: {args.version}")

    if args.ckpt_path is None:
        args.ckpt_path = ckpt_path

    if args.scale is None:
        args.scale = 9.0 if args.version.startswith("2.") else 7.5

    logger = set_logger(
        name="Image-to-Image with Adapter",
        output_dir=args.output_path,
        rank=0,
    )

    # core task
    main(args)
