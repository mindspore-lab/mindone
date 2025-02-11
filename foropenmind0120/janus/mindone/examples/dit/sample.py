import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np
import yaml
from PIL import Image
from utils.model_utils import _check_cfgs_in_parser, count_params, load_dit_ckpt_params, remove_pname_prefix, str2bool
from utils.plot import image_grid

import mindspore as ms
from mindspore import Tensor, ops

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from modules.autoencoder import SD_CONFIG, AutoencoderKL

from examples.dit.pipelines.infer_pipeline import DiTInferPipeline
from mindone.models.dit import DiT_models
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.mode,
        device_target=args.device_target,
        device_id=device_id,
    )
    if args.mode == ms.GRAPH_MODE:
        try:
            if args.jit_level in ["O0", "O1", "O2"]:
                ms.set_context(jit_config={"jit_level": args.jit_level})
                logger.info(f"set jit_level: {args.jit_level}.")
            else:
                logger.warning(
                    f"Unsupport jit_level: {args.jit_level}. The framework automatically selects the execution method"
                )
        except Exception:
            logger.warning(
                "The current jit_level is not suitable because current MindSpore version does not match,"
                "please ensure the MindSpore version >= ms2.3.0."
            )
    if args.precision_mode is not None:
        ms.set_context(ascend_config={"precision_mode": args.precision_mode})

    return device_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="image size",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="DiT-XL/2",
        help="Model name , such as DiT-XL/2, DiT-L/2",
    )
    parser.add_argument(
        "--dit_checkpoint", type=str, default="models/DiT-XL-2-256x256.ckpt", help="the path to the DiT checkpoint."
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-mse.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )
    parser.add_argument("--sampling_steps", type=int, default=50, help="Diffusion Sampling Steps")
    parser.add_argument("--guidance_scale", type=float, default=8.5, help="the scale for classifier-free guidance")
    # MS new args
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    parser.add_argument(
        "--use_recompute",
        default=None,
        type=str2bool,
        help="whether use recompute.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument(
        "--patch_embedder",
        type=str,
        default="conv",
        choices=["conv", "linear"],
        help="Whether to use conv2d layer or dense (linear layer) as Patch Embedder.",
    )
    parser.add_argument("--ddim_sampling", type=str2bool, default=True, help="Whether to use DDIM for sampling")
    parser.add_argument("--imagegrid", default=False, type=str2bool, help="Save the image in image-grids format.")
    parser.add_argument(
        "--jit_level",
        default="O2",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports [“O0”, “O1”, “O2”]."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    args = parse_args()
    init_env(args)
    set_random_seed(args.seed)

    # 2. model initiate and weight loading
    # 2.1 dit
    logger.info(f"{args.model_name}-{args.image_size}x{args.image_size} init")
    latent_size = args.image_size // 8
    dit_model = DiT_models[args.model_name](
        input_size=latent_size,
        num_classes=1000,
        block_kwargs={"enable_flash_attention": args.enable_flash_attention},
        patch_embedder=args.patch_embedder,
        use_recompute=args.use_recompute,
    )

    if args.dtype == "fp16":
        model_dtype = ms.float16
        dit_model = auto_mixed_precision(dit_model, amp_level="O2", dtype=model_dtype)
    elif args.dtype == "bf16":
        model_dtype = ms.bfloat16
        dit_model = auto_mixed_precision(dit_model, amp_level="O2", dtype=model_dtype)
    else:
        model_dtype = ms.float32

    try:
        dit_model = load_dit_ckpt_params(dit_model, args.dit_checkpoint)
    except Exception:
        param_dict = ms.load_checkpoint(args.dit_checkpoint)
        param_dict = remove_pname_prefix(param_dict, prefix="network.")
        dit_model = load_dit_ckpt_params(dit_model, param_dict)
    dit_model = dit_model.set_train(False)
    for param in dit_model.get_parameters():  # freeze dit_model
        param.requires_grad = False

    # 2.2 vae
    logger.info("vae init")
    vae = AutoencoderKL(
        SD_CONFIG,
        4,
        ckpt_path=args.vae_checkpoint,
        use_fp16=False,  # disable amp for vae
    )
    vae = vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # Create sampling noise:
    n = len(class_labels)
    z = ops.randn((n, 4, latent_size, latent_size), dtype=ms.float32)
    y = Tensor(class_labels)
    y_null = ops.ones_like(y) * 1000

    # 3. build inference pipeline
    pipeline = DiTInferPipeline(
        dit_model,
        vae,
        scale_factor=args.sd_scale_factor,
        num_inference_steps=args.sampling_steps,
        guidance_rescale=args.guidance_scale,
        ddim_sampling=args.ddim_sampling,
    )

    # 4. print key info
    num_params_vae, num_params_vae_trainable = count_params(vae)
    num_params_dit, num_params_dit_trainable = count_params(dit_model)
    num_params = num_params_vae + num_params_dit
    num_params_trainable = num_params_vae_trainable + num_params_dit_trainable
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Class labels: {class_labels}",
            f"Num params: {num_params:,} (dit: {num_params_dit:,}, vae: {num_params_vae:,})",
            f"Num trainable params: {num_params_trainable:,}",
            f"Use model dtype: {model_dtype}",
            f"Sampling steps {args.sampling_steps}",
            f"DDIM sampling: {args.ddim_sampling}",
            f"CFG guidance scale: {args.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)
    # init inputs
    inputs = {}
    inputs["noise"] = z
    inputs["y"] = y
    inputs["y_null"] = y_null
    inputs["scale"] = args.guidance_scale

    logger.info(f"Sampling class labels: {class_labels}")
    start_time = time.time()

    # infer
    x_samples = pipeline(inputs)
    x_samples = x_samples.asnumpy()

    end_time = time.time()

    # save result
    if not args.imagegrid:
        for i, class_label in enumerate(class_labels, 0):
            save_fp = f"{save_dir}/class-{class_label}.png"
            img = Image.fromarray((x_samples[i] * 255).astype(np.uint8))
            img.save(save_fp)
            logger.info(f"save to {save_fp}")
    else:
        save_fp = f"{save_dir}/sample.png"
        img = image_grid(x_samples)
        img.save(save_fp)
        logger.info(f"save to {save_fp}")
