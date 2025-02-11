import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
from adapters import CombinedAdapter, get_adapter
from jsonargparse import ActionConfigFile, ArgumentParser
from omegaconf import OmegaConf
from PIL import Image
from t2i_utils import read_images

import mindspore as ms

sys.path.append("../../")  # FIXME: remove in future when mindone is ready for install
from mindone.utils import set_logger
from mindone.utils.env import init_train_env

sys.path.append("../stable_diffusion_xl/")
from gm.helpers import SD_XL_BASE_RATIOS, VERSION2SPECS, create_model, init_sampling


def prepare_infer_dict(
    resolution: Tuple[int, int],
    crop_coords_top_left: Tuple[int, int] = (0, 0),
    aesthetic_score: float = 6.0,
    negative_aesthetic_score: float = 2.5,
):
    return {
        "orig_height": resolution[0],
        "orig_width": resolution[1],
        "target_height": resolution[0],
        "target_width": resolution[1],
        "crop_coords_top": crop_coords_top_left[0],
        "crop_coords_left": crop_coords_top_left[1],
        "aesthetic_score": aesthetic_score,
        "negative_aesthetic_score": negative_aesthetic_score,
    }


def main(args):
    # read prompts and conditions
    prompts = args.prompt
    negative_prompts = args.negative_prompt if args.negative_prompt else [""] * len(prompts)

    if len(args.adapter.cond_weight) == 1:  # if condition weights are not specified per adapter
        args.adapter.cond_weight *= len(args.adapter.condition)
    cond_weights = args.adapter.cond_weight

    assert len(args.adapter.condition) == len(args.adapter.ckpt_path) == len(args.adapter.image) == len(cond_weights), (
        f"Number of adapters and conditions should match, got {args.adapter_condition} adapters,"
        f" {args.adapter_ckpt_path} checkpoints, {args.image} condition images, and {cond_weights} weights."
    )

    cond_paths = args.adapter.image
    assert all([os.path.isfile(cond) for cond in cond_paths]), "Paths to condition images must be files."

    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    sample_path = output_path / "sdxl" / "_".join(args.adapter.condition) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sample_path.mkdir(exist_ok=True, parents=True)

    # set ms context
    init_train_env(**args.environment)
    if args.environment.mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": args.jit_level})
    # initialize SD and adapter models
    args.SDXL.config = OmegaConf.load(args.SDXL.config)  # NOQA
    overwrite = args.SDXL.pop("overwrite", {})
    if overwrite:
        args.SDXL.config = OmegaConf.merge(args.SDXL.config, overwrite)

    version = args.SDXL.config.pop("version", "SDXL-base-1.0")
    model_ratio = args.SDXL.pop("ratio", "1.0")

    model, _ = create_model(**args.SDXL, freeze=True)

    adapters = [
        get_adapter("sdxl", a_cond, ckpt, use_fp16=False)
        for a_cond, ckpt in zip(args.adapter.condition, args.adapter.ckpt_path)
    ]
    adapters = CombinedAdapter(adapters, cond_weights, output_fp16=args.SDXL.amp_level == "O2")

    # create sampler
    sampler, *_ = init_sampling(**args.sampler)

    # log
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.environment.mode}",
            f"Prompt: {prompts}",
            f"Negative prompt: {negative_prompts}",
            f"Adapter conditions: {args.adapter.condition}",
            f"Adapter checkpoints: {args.adapter.ckpt_path}",
            f"Adapter condition weights: {args.adapter.cond_weight}",
            f"Condition images: {cond_paths}",
            f"Number of samples in each trial: {args.n_samples}",
            f"Model: {version}",
            f"Precision: {'FP16' if args.SDXL.amp_level == 'O2' else 'FP32'}",
            f"Pretrained ckpt path: {args.SDXL.checkpoints}",
            f"Sampler: {args.sampler.sampler}",
            f"Sampling steps: {args.sampler.steps}",
            f"Unconditional guidance scale: {args.sampler.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    w, h = SD_XL_BASE_RATIOS[model_ratio]
    version_dict = VERSION2SPECS.get(version)
    c, f = version_dict["C"], version_dict["f"]

    # infer
    flags = [-1 for _ in range(len(args.adapter.condition))]
    conds, img_shape = read_images(cond_paths, min(h, w), flags=flags)
    adapter_features, _ = adapters(conds)

    value_dict = prepare_infer_dict(
        img_shape, args.crop_coords_top_left, args.aesthetic_score, args.negative_aesthetic_score
    )

    base_count = 0
    for i, (prompt, negative_prompt) in enumerate(zip(prompts, negative_prompts)):
        for n in range(args.n_iter):
            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt

            start_time = time.perf_counter()
            out = model.do_sample(
                sampler,
                value_dict,
                args.n_samples,
                *img_shape,
                c,
                f,
                adapter_states=tuple(adapter_features) if isinstance(adapter_features, list) else adapter_features,
                amp_level=args.SDXL.amp_level,
            )

            for sample in out:
                sample = 255.0 * sample.transpose(1, 2, 0)
                Image.fromarray(sample.astype(np.uint8)).save(sample_path / f"{base_count:05}.png")
                base_count += 1

            logger.info(
                f"{args.n_samples * (i + 1) * (n + 1)}/{args.n_samples * len(prompts) * args.n_iter} images generated, "
                f"time cost for current trial: {time.perf_counter() - start_time:.3f}s"
            )

    logger.info(f"Done! All generated images are saved in: {output_path}\nEnjoy.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(
        init_train_env, "environment", skip={"distributed", "enable_modelarts", "num_workers", "json_data_path"}
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
    # Stable Diffusion
    parser.add_function_arguments(create_model, "SDXL", skip={"config", "freeze", "load_filter"})
    parser.add_argument("--SDXL.config", type=str, default="configs/inference/sd_xl_base.yaml")
    parser.add_argument("--SDXL.overwrite", type=dict, help="Parameters to overwrite in SDXL config.")
    parser.add_argument("--SDXL.ratio", type=str, default="1.0")
    parser.add_function_arguments(
        init_sampling, "sampler", skip={"num_cols", "specify_num_samples", "img2img_strength", "stage2strength"}
    )

    # T2I-Adapter
    parser.add_argument(
        "--adapter.condition",
        type=str,
        nargs="+",
        help="Additional condition(s) for spatial (visual) guidance of SD with support of adapter."
        "Allows passing multiple conditions for Combined Adapters.",
    )
    parser.add_argument(
        "--adapter.ckpt_path",
        type=str,
        nargs="+",
        help="Path(s) to the adapter checkpoint(s).",
    )
    parser.add_argument(
        "--adapter.cond_weight",
        type=float,
        nargs="+",
        default=[1.0],
        help="The adapter features are multiplied by the `cond_weight`. The larger the `cond_weight`, the more aligned "
        "the generated image and condition will be, but the generated quality may be reduced.",
    )
    parser.add_argument(
        "--adapter.image",
        type=str,
        nargs="+",
        help="Path(s) to a condition image(s).",
    )

    # Image generation
    parser.add_argument("--prompt", type=str, nargs="+", help="added prompt")
    parser.add_argument("--negative_prompt", type=str, nargs="*", help="the negative prompt not to render")
    parser.add_argument("--resolution", type=int, help="target image resolution")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="number of iterations or trials. sample this often, ",
    )
    parser.add_function_arguments(prepare_infer_dict, skip={"resolution"})

    parser.add_argument("--output_path", type=str, default="output", help="dir to write results to")

    args = parser.parse_args()

    logger = set_logger(
        name="Text-to-Image with Adapter",
        output_dir=args.output_path,
        rank=0,
    )

    # core task
    main(args)
