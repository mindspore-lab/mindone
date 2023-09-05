# reference to https://github.com/Stability-AI/generative-models
import argparse
import ast
import os

from gm.helpers import SD_XL_BASE_RATIOS, VERSION2SPECS, create_model, init_sampling, perform_save_locally
from gm.util import seed_everything
from omegaconf import OmegaConf

import mindspore as ms


def get_parser_sample():
    parser = argparse.ArgumentParser(description="sampling with sd-xl")
    parser.add_argument("--version", type=str, default="SDXL-base-1.0")
    parser.add_argument("--task", type=str, default="txt2img", choices=["txt2img", "img2img"])
    parser.add_argument("--config", type=str, default="configs/inference/sd_xl_base.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="outputs/demo/txt2img/", help="save dir")
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
    parser.add_argument("--sampler", type=str, default="EulerEDMSampler")
    parser.add_argument("--guider", type=str, default="VanillaCFG")
    parser.add_argument("--discretization", type=str, default="LegacyDDPMDiscretization")
    parser.add_argument("--weight", type=str, default="checkpoints/sd_xl_base_1.0_ms.ckpt")
    parser.add_argument("--sample_step", type=int, default=40)
    parser.add_argument("--num_cols", type=int, default=1)

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
        "aesthetic_score": 6.0,
        "negative_aesthetic_score": 2.5,
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
    return out


def sample(args):
    assert args.version in VERSION2SPECS
    version_dict = VERSION2SPECS.get(args.version)
    mode = args.task

    add_pipeline = False
    # TODO: Add Refiner
    # if version.startswith("SDXL-base"):
    #     add_pipeline = st.checkbox("Load SDXL-refiner?", False)
    #     st.write("__________________________")

    seed_everything(args.seed)

    # Init Model
    config = OmegaConf.load(args.config)
    model, filter = create_model(
        config, checkpoints=args.weight.split(","), freeze=True, load_filter=False, amp_level=args.ms_amp_level
    )  # TODO: Add filter support

    save_path = os.path.join(args.save_path, args.version)
    is_legacy = version_dict["is_legacy"]
    args.negative_prompt = args.negative_prompt if is_legacy else ""

    if add_pipeline:
        # TODO: Add Refiner
        raise NotImplementedError

    if mode == "txt2img":
        out = run_txt2img(
            args,
            model,
            version_dict,
            is_legacy=is_legacy,
            return_latents=add_pipeline,
            filter=filter,
            stage2strength=None,
            amp_level=args.ms_amp_level,
        )
    elif mode == "img2img":
        raise NotImplementedError
    else:
        raise ValueError(f"unknown mode {mode}")

    out = out if isinstance(out, (tuple, list)) else [out, None]
    (samples, samples_z) = out

    if add_pipeline and samples_z is not None:
        raise NotImplementedError

    assert samples is not None
    perform_save_locally(save_path, samples)


if __name__ == "__main__":
    parser = get_parser_sample()
    args, _ = parser.parse_known_args()
    ms.context.set_context(mode=args.ms_mode, device_target=args.device_target)
    sample(args)
