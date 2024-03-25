"""
Text to image generation
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
from common import init_env
from omegaconf import OmegaConf
from PIL import Image

import mindspore as ms

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workspace)
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.uni_pc import UniPCSampler
from ldm.modules.logger import set_logger
from ldm.modules.lora import inject_trainable_lora, inject_trainable_lora_to_textencoder
from ldm.util import instantiate_from_config, str2bool
from tools.safety_checker import SafetyChecker
from tools.watermark import WatermarkEmbedder
from utils import model_utils
from utils.download import download_checkpoint
from utils.long_prompt import get_text_embeddings

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
CLIP_CKPT_URL = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/clip/clip_vit_l_14.ckpt"
_MIN_CKPT_SIZE = 4.0 * 1e9


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(
    config, ckpt, use_lora=False, lora_rank=4, lora_fp16=True, lora_only_ckpt=None, ti_only_ckpt=None
):
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

    if use_lora:
        load_lora_only = True if lora_only_ckpt is not None else False
        if not load_lora_only:
            logger.info(f"Loading model from {ckpt}")
            _load_model(model, ckpt)
        else:
            if os.path.exists(lora_only_ckpt):
                lora_param_dict = ms.load_checkpoint(lora_only_ckpt)
                if "lora_rank" in lora_param_dict.keys():
                    lora_rank = int(lora_param_dict["lora_rank"].value())
                    logger.info(f"Lora rank is set to {lora_rank} according to the found value in lora checkpoint.")
            else:
                raise ValueError(f"{lora_only_ckpt} doesn't exist")
            # load the main pretrained model
            logger.info(f"Loading pretrained model from {ckpt}")
            _load_model(model, ckpt, verbose=True, filter=ms.load_checkpoint(ckpt).keys())
            # inject lora params
            if args.lora_ft_unet:
                injected_attns, injected_trainable_params = inject_trainable_lora(
                    model,
                    rank=lora_rank,
                    use_fp16=(model.model.diffusion_model.dtype == ms.float16),
                    scale=args.lora_scale,
                )
            if args.lora_ft_text_encoder:
                injected_attns, injected_trainable_params = inject_trainable_lora_to_textencoder(
                    model,
                    rank=lora_rank,
                    use_fp16=(model.model.diffusion_model.dtype == ms.float16),
                    scale=args.lora_scale,
                )

            # load fine-tuned lora params
            logger.info(f"Loading LoRA params from {lora_only_ckpt}")
            _load_model(model, lora_only_ckpt, verbose=True, filter=injected_trainable_params.keys())
    else:
        logger.info(f"Loading model from {ckpt}")
        _load_model(model, ckpt)
    if ti_only_ckpt is not None:
        from ldm.modules.textual_inversion.manager import TextualInversionManager

        logger.info(f"Loading Textual Inversion params from {ti_only_ckpt}")
        manager = TextualInversionManager(
            model,
        )
        manager.load_checkpoint_textual_inversion(ti_only_ckpt)

    model.set_train(False)
    for param in model.trainable_params():
        param.requires_grad = False
    if ti_only_ckpt is not None:
        return model, manager
    return model


def main(args):
    # init
    device_id, rank_id, device_num = init_env(
        args.ms_mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device_target,
    )

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

    # read prompts
    batch_size = args.n_samples
    if not args.data_path:
        prompt = args.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        logger.info(f"Reading prompts from {args.data_path}")
        with open(args.data_path, "r") as f:
            prompts = f.read().splitlines()
            # TODO: try to put different prompts in a batch
            data = [batch_size * [prompt] for prompt in prompts]

    # read negative prompts
    if not args.negative_data_path:
        negative_prompt = args.negative_prompt
        assert negative_prompt is not None
        negative_data = [batch_size * [negative_prompt]]
    else:
        logger.info(f"Reading negative prompts from {args.negative_data_path}")
        with open(args.negative_data_path, "r") as f:
            negative_prompts = f.read().splitlines()
            # TODO: try to put different prompts in a batch
            negative_data = [batch_size * [negative_prompt] for negative_prompt in negative_prompts]

    # post-process negative prompts
    assert len(negative_data) <= len(data), "Negative prompts should be shorter than positive prompts"
    if len(negative_data) < len(data):
        logger.info("Negative prompts are shorter than positive prompts, padding blank prompts")
        blank_negative_prompt = batch_size * [""]
        for _ in range(len(data) - len(negative_data)):
            negative_data.append(blank_negative_prompt)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path)) + rank_id * args.n_samples * args.n_iter

    # create model
    if not os.path.isabs(args.config):
        args.config = os.path.join(work_dir, args.config)
    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(
        config,
        ckpt=args.ckpt_path,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_only_ckpt=args.lora_ckpt_path,
        ti_only_ckpt=args.ti_ckpt_path,
    )
    if args.ti_ckpt_path is not None:
        model, manager = model
        data = [[manager.manage_prompt(p) for p in prompts] for prompts in data]

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
            "ddim",
            "dpm_solver",
            "dpm_solver_pp",
        ], "Only ddim, dpm_solver and dpm_solver_pp support v-prediction currently."

    # create safety checker
    if args.check_safety:
        if args.clip_ckpt_path is None:
            clip_ckpt_name = os.path.basename(CLIP_CKPT_URL)
            args.clip_ckpt_path = "models/" + clip_ckpt_name
            if not os.path.exists(args.clip_ckpt_path):
                print(f"Start downloading checkpoint {clip_ckpt_name} ...")
                download_checkpoint(CLIP_CKPT_URL, "models/")

        safety_checker = SafetyChecker(safety_version=args.safety_version, backend="ms", ckpt_path=args.clip_ckpt_path)

    # log
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
            f"Distributed mode: {args.use_parallel}",
            f"Number of input prompts: {len(data)}",
            f"Number of input negative prompts: {len(negative_data)}",
            f"Number of trials for each prompt: {args.n_iter}",
            f"Number of samples in each trial: {args.n_samples}",
            f"Model: StableDiffusion v-{args.version}",
            f"Precision: {model.model.diffusion_model.dtype}",
            f"Pretrained ckpt path: {args.ckpt_path}",
            f"Lora ckpt path: {args.lora_ckpt_path if args.use_lora else None}",
            f"Textual Inversion ckpt path: {args.ti_ckpt_path}",
            f"Sampler: {sname}",
            f"Sampling steps: {args.sampling_steps}",
            f"Uncondition guidance scale: {args.scale}",
            f"Target image size (H, W): ({args.H}, {args.W})",
        ]
    )
    key_info += "\n" + "=" * 50
    if rank_id == 0:
        logger.info(key_info)

    # infer
    start_code = None
    if args.fixed_code:
        stdnormal = ms.ops.StandardNormal()
        start_code = stdnormal((args.n_samples, 4, args.H // 8, args.W // 8))

    all_samples = list()
    for i in range(0, len(data), device_num):
        if i > 0:
            base_count += args.n_samples * args.n_iter * (device_num - 1)
        if i + rank_id < len(data):
            prompts = data[i + rank_id]
            negative_prompts = negative_data[i + rank_id]
        else:
            break

        logger.info(
            f"[{i + rank_id + 1}/{len(data)}] Generating images with conditions:\nPrompt(s): {prompts[0]}\n"
            f"Negative prompt(s): {negative_prompts[0]}"
        )
        for n in range(args.n_iter):
            start_time = time.time()
            uc = None
            if args.scale != 1.0:
                if isinstance(negative_prompts, tuple):
                    negative_prompts = list(negative_prompts)
            else:
                negative_prompts = None
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            c, uc = get_text_embeddings(
                model, prompts, negative_prompts, support_long_prompts=args.support_long_prompts
            )
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
            x_samples_ddim = ms.ops.clip_by_value((x_samples_ddim + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

            if args.add_watermark:
                water_mark = WatermarkEmbedder(dtype=model.model.diffusion_model.dtype)
                x_samples_ddim = water_mark(x_samples_ddim)
            if args.check_safety:
                x_samples_ddim, _ = safety_checker(x_samples_ddim)

            x_samples_ddim_numpy = x_samples_ddim.asnumpy()

            if not args.skip_save:
                for x_sample in x_samples_ddim_numpy:
                    x_sample = 255.0 * x_sample.transpose(1, 2, 0)
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1

            if not args.skip_grid:
                all_samples.append(x_samples_ddim_numpy)

            end_time = time.time()
            logger.info(
                f"{batch_size * (n + 1)}/{batch_size * args.n_iter} images generated, "
                f"time cost for current trial: {end_time - start_time:.3f}s"
            )
    if rank_id == 0:
        logger.info(f"Done! All generated images are saved in: {outpath}/samples" f"\nEnjoy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--device_target", type=str, nargs="?", default="Ascend", help="Ascend, GPU")
    parser.add_argument(
        "--data_path",
        type=str,
        nargs="?",
        default="",
        help="path to a file containing prompt list (each line in the file corresponds to a prompt to render).",
    )
    parser.add_argument(
        "--negative_data_path",
        type=str,
        nargs="?",
        default="",
        help="path to a file containing negative prompt list (each line in the file corresponds to a prompt not to "
        "render).",
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
        "--prompt", type=str, nargs="?", default="A cute wolf in winter forest", help="the prompt to render"
    )
    parser.add_argument("--negative_prompt", type=str, nargs="?", default="", help="the negative prompt not to render")
    parser.add_argument("--output_path", type=str, nargs="?", default="output", help="dir to write results to")
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )
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
        default=4,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
    parser.add_argument(
        "--support_long_prompts",
        default=False,
        type=str2bool,
        help="Whether to support long prompts exceeding the context length. If False, it will truncate the text prompts",
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
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="unconditional guidance scale: eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
        "Simplified: `uc + scale * (uc - prompt)`",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to config which constructs model. If None, select by version",
    )
    parser.add_argument(
        "--use_lora",
        default=False,
        type=str2bool,
        help="whether the checkpoint used for inference is finetuned from LoRA",
    )
    parser.add_argument("--lora_ft_unet", default=True, type=str2bool, help="whether lora finetune is applied to unet")
    parser.add_argument(
        "--lora_ft_text_encoder", default=False, type=str2bool, help="whether lora finetune is applied to text encoder"
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
        "--ti_ckpt_path",
        type=str,
        default=None,
        help="path to textual inversion only checkpoint. ",
    )
    parser.add_argument(
        "--lora_scale",
        default=1.0,
        type=float,
        help="scale, the higher, the more LoRA weights will affect orignal SD. If 0, LoRA has no effect.",
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
    parser.add_argument(
        "--add_watermark",
        action="store_true",
        help="whether add invisible watermark to image",
    )
    parser.add_argument(
        "--check_safety",
        action="store_true",
        help="set this flag to use a safety checker",
    )
    parser.add_argument(
        "--clip_ckpt_path",
        type=str,
        default=None,
        help="path to checkpoint of clip-vit-large-patch14 for safety checker",
    )
    parser.add_argument(
        "--safety_version",
        type=int,
        default=2,
        help="the version of stable diffusion to use for its safety checker. Option: 1, 2" "Default: 2",
    )
    args = parser.parse_args()

    # check args
    if args.version:
        if args.version not in _version_cfg:
            raise ValueError(f"Unknown version: {args.version}. Supported SD versions are: {list(_version_cfg.keys())}")
    if args.ckpt_path is None:
        ckpt_name = _version_cfg[args.version][0]
        args.ckpt_path = "models/" + ckpt_name

        desire_size = _version_cfg[args.version][2]
        if args.H != desire_size or args.W != desire_size:
            logger.warning(
                f"The optimal H, W for SD {args.version} is ({desire_size}, {desire_size}) . But got ({args.H}, {args.W})."
            )

        # download if not exists or not complete
        ckpt_incomplete = False
        if os.path.exists(args.ckpt_path):
            if os.path.getsize(args.ckpt_path) < _MIN_CKPT_SIZE:
                ckpt_incomplete = True
                print(
                    f"WARNING: The checkpoint size is too small {args.ckpt_path}. Please check and remove it if it is incomplete!"
                )
        if not os.path.exists(args.ckpt_path):
            print(f"Start downloading checkpoint {ckpt_name} ...")
            download_checkpoint(os.path.join(_URL_PREFIX, ckpt_name), "models/")

    if args.config is None:
        args.config = os.path.join("configs", _version_cfg[args.version][1])

    if args.scale is None:
        args.scale = 9.0 if args.version.startswith("2.") else 7.5

    # core task
    main(args)
