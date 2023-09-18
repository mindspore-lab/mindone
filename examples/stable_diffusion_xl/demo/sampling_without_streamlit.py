# reference to https://github.com/Stability-AI/generative-models
import argparse
import ast
import os
import time

from gm.helpers import SD_XL_BASE_RATIOS, VERSION2SPECS, create_model, init_sampling, load_img, perform_save_locally
from gm.util import seed_everything
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import Tensor, ops


def get_parser_sample():
    parser = argparse.ArgumentParser(description="sampling with sd-xl")
    parser.add_argument("--task", type=str, default="txt2img", choices=["txt2img", "img2img"])
    parser.add_argument("--config", type=str, default="configs/inference/sd_xl_base.yaml")
    parser.add_argument("--weight", type=str, default="checkpoints/sd_xl_base_1.0_ms.ckpt")
    parser.add_argument(
        "--prompt", type=str, default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--sd_xl_base_ratios", type=str, default="1.0")
    parser.add_argument("--orig_width", type=int, default=None)
    parser.add_argument("--orig_height", type=int, default=None)
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--crop_coords_top", type=int, default=None)
    parser.add_argument("--crop_coords_left", type=int, default=None)
    parser.add_argument("--aesthetic_score", type=float, default=None)
    parser.add_argument("--negative_aesthetic_score", type=float, default=None)
    parser.add_argument("--sampler", type=str, default="EulerEDMSampler")
    parser.add_argument("--guider", type=str, default="VanillaCFG")
    parser.add_argument("--discretization", type=str, default="LegacyDDPMDiscretization")
    parser.add_argument("--sample_step", type=int, default=40)
    parser.add_argument("--num_cols", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="outputs/demo/", help="save dir")

    # for img2img
    parser.add_argument("--img", type=str, default=None)
    parser.add_argument("--strength", type=float, default=0.75)

    # for pipeline
    parser.add_argument("--add_pipeline", type=ast.literal_eval, default=False)
    parser.add_argument("--pipeline_config", type=str, default="configs/inference/sd_xl_refiner.yaml")
    parser.add_argument("--pipeline_weight", type=str, default="checkpoints/sd_xl_refiner_1.0_ms.ckpt")
    parser.add_argument("--stage2strength", type=float, default=0.15)
    parser.add_argument("--finish_denoising", type=ast.literal_eval, default=True)

    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument(
        "--ms_mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)"
    )
    parser.add_argument("--ms_jit", type=ast.literal_eval, default=True, help="use jit or not")
    parser.add_argument("--ms_amp_level", type=str, default="O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )

    # args for ModelArts
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument(
        "--ckpt_url", type=str, default="", help="ModelArts: obs path to pretrain model checkpoint file"
    )
    parser.add_argument("--train_url", type=str, default="", help="ModelArts: obs path to output folder")
    parser.add_argument(
        "--multi_data_url", type=str, default="", help="ModelArts: list of obs paths to multi-dataset folders"
    )
    parser.add_argument(
        "--pretrain_url", type=str, default="", help="ModelArts: list of obs paths to multi-pretrain model files"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/cache/pretrain_ckpt/",
        help="ModelArts: local device path to checkpoint folder",
    )
    return parser


def run_txt2img(
    args, model, version_dict, is_legacy=False, return_latents=False, filter=None, stage2strength=None, amp_level="O0"
):
    assert args.sd_xl_base_ratios in SD_XL_BASE_RATIOS
    W, H = SD_XL_BASE_RATIOS[args.sd_xl_base_ratios]
    C = version_dict["C"]
    F = version_dict["f"]

    value_dict = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "orig_width": args.orig_width if args.orig_width else W,
        "orig_height": args.orig_height if args.orig_height else H,
        "target_width": args.target_width if args.target_width else W,
        "target_height": args.target_height if args.target_height else H,
        "crop_coords_top": max(args.crop_coords_top if args.crop_coords_top else 0, 0),
        "crop_coords_left": max(args.crop_coords_left if args.crop_coords_left else 0, 0),
        "aesthetic_score": args.aesthetic_score if args.aesthetic_score else 6.0,
        "negative_aesthetic_score": args.negative_aesthetic_score if args.negative_aesthetic_score else 2.5,
    }
    sampler, num_rows, num_cols = init_sampling(
        sampler=args.sampler,
        num_cols=args.num_cols,
        guider=args.guider,
        discretization=args.discretization,
        steps=args.sample_step,
        stage2strength=stage2strength,
    )
    num_samples = num_rows * num_cols

    print("Txt2Img Sampling")
    s_time = time.time()
    out = model.do_sample(
        sampler,
        value_dict,
        num_samples,
        H,
        W,
        C,
        F,
        force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        return_latents=return_latents,
        filter=filter,
        amp_level=amp_level,
    )
    print(f"Txt2Img sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")

    return out


def run_img2img(args, model, is_legacy=False, return_latents=False, filter=None, stage2strength=None, amp_level="O0"):
    dtype = ms.float32 if amp_level not in ("O2", "O3") else ms.float16

    img = load_img(args.img)
    assert img is not None
    H, W = img.shape[2], img.shape[3]

    value_dict = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "orig_width": args.orig_width if args.orig_width else W,
        "orig_height": args.orig_height if args.orig_height else H,
        "target_width": args.target_width if args.target_width else W,
        "target_height": args.target_height if args.target_height else H,
        "crop_coords_top": max(args.crop_coords_top if args.crop_coords_top else 0, 0),
        "crop_coords_left": max(args.crop_coords_left if args.crop_coords_left else 0, 0),
        "aesthetic_score": args.aesthetic_score if args.aesthetic_score else 6.0,
        "negative_aesthetic_score": args.negative_aesthetic_score if args.negative_aesthetic_score else 2.5,
    }
    strength = min(max(args.strength, 0.0), 1.0)
    print("**Img2Img Strength**: strength")
    sampler, num_rows, num_cols = init_sampling(
        sampler=args.sampler,
        num_cols=args.num_cols,
        guider=args.guider,
        discretization=args.discretization,
        steps=args.sample_step,
        img2img_strength=strength,
        stage2strength=stage2strength,
    )
    num_samples = num_rows * num_cols

    print("Img2Img Sampling")
    s_time = time.time()
    out = model.do_img2img(
        ops.repeat_elements(Tensor(img, dtype), num_samples, axis=0),
        sampler,
        value_dict,
        num_samples,
        force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        return_latents=return_latents,
        filter=filter,
        amp_level=amp_level,
    )
    print(f"Img2Img sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")

    return out


def apply_refiner(
    input, model, sampler, num_samples, prompt, negative_prompt, filter=None, finish_denoising=False, amp_level="O0"
):
    latent_h, latent_w = input.shape[2:]
    value_dict = {
        "orig_width": latent_w * 8,
        "orig_height": latent_h * 8,
        "target_width": latent_w * 8,
        "target_height": latent_h * 8,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "crop_coords_top": 0,
        "crop_coords_left": 0,
        "aesthetic_score": 6.0,
        "negative_aesthetic_score": 2.5,
    }

    print("Img2Img Sampling")
    print(f"WARNING: refiner input shape: {input.shape}")
    s_time = time.time()
    samples = model.do_img2img(
        input,
        sampler,
        value_dict,
        num_samples,
        skip_encode=True,
        filter=filter,
        add_noise=not finish_denoising,
        amp_level=amp_level,
    )
    print(f"PipeLine(Refiner) sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")

    return samples


def sample(args):
    config = OmegaConf.load(args.config)
    version = config.pop("version", "SDXL-base-1.0")
    version_dict = VERSION2SPECS.get(version)

    task = args.task

    add_pipeline = args.add_pipeline
    if not version.startswith("SDXL-base") and add_pipeline:
        add_pipeline = False
        print(
            f"'add_pipeline' is only supported on SDXL-base model, but got {version}, 'add_pipeline' modify to 'False'"
        )

    seed_everything(args.seed)

    # Init Model
    model, filter = create_model(
        config, checkpoints=args.weight.split(","), freeze=True, load_filter=False, amp_level=args.ms_amp_level
    )  # TODO: Add filter support

    save_path = os.path.join(args.save_path, task, version)
    is_legacy = version_dict["is_legacy"]
    args.negative_prompt = args.negative_prompt if is_legacy else ""

    stage2strength = None

    if add_pipeline:
        # Init for pipeline
        version2 = "SDXL-refiner-1.0"
        config2 = args.pipeline_config
        weight2 = args.pipeline_weight
        stage2strength = args.stage2strength
        print(f"WARNING: Running with {version2} as the second stage model. Make sure to provide (V)RAM :) ")
        if args.device_target == "Ascend" and args.sd_xl_base_ratios not in ("1.0_768", "1.0_512"):
            print(
                "Warning: Using the 'add_pipeline' function on device Ascend 910A may cause OOM. "
                "It is recommended to use txt2img and img2img tasks respectively. "
                "Alternatively, use smaller generation sizes such as (768, 768) "
                "by configuring `--sd_xl_base_ratios '1.0_768'`."
            )

        # Init Model
        model2, filter2 = create_model(
            config2, checkpoints=weight2.split(","), freeze=True, load_filter=False, amp_level=args.ms_amp_level
        )

        stage2strength = min(max(stage2strength, 0.0), 1.0)
        print(f"**Refinement strength**: {stage2strength}")

        sampler2, *_ = init_sampling(
            sampler=args.sampler,
            num_cols=args.num_cols,
            guider=args.guider,
            discretization=args.discretization,
            steps=args.sample_step,
            img2img_strength=stage2strength,
            specify_num_samples=False,
        )
        if not args.finish_denoising:
            stage2strength = None

    if task == "txt2img":
        out = run_txt2img(
            args,
            model,
            version_dict,
            is_legacy=is_legacy,
            return_latents=add_pipeline,
            filter=filter,
            stage2strength=stage2strength,
            amp_level=args.ms_amp_level,
        )
    elif task == "img2img":
        out = run_img2img(
            args,
            model,
            is_legacy=is_legacy,
            return_latents=add_pipeline,
            filter=filter,
            stage2strength=stage2strength,
            amp_level=args.amp_level,
        )
    else:
        raise ValueError(f"Unknown task {task}")

    out = out if isinstance(out, (tuple, list)) else [out, None]
    (samples, samples_z) = out

    perform_save_locally(save_path, samples)

    if add_pipeline:
        print("**Running Refinement Stage**")
        assert samples_z is not None

        samples = apply_refiner(
            samples_z,
            model=model2,
            sampler=sampler2,
            num_samples=samples_z.shape[0],
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if is_legacy else "",
            filter=filter2,
            finish_denoising=args.finish_denoising,
        )

        perform_save_locally(os.path.join(save_path, "pipeline"), samples)


if __name__ == "__main__":
    parser = get_parser_sample()
    args, _ = parser.parse_known_args()
    ms.context.set_context(mode=args.ms_mode, device_target=args.device_target)
    sample(args)
