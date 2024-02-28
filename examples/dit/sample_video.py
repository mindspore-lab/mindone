import argparse
import datetime
import logging
import os
import sys
import time

import yaml
from utils.model_utils import _check_cfgs_in_parser, count_params, str2bool

import mindspore as ms
from mindspore import Tensor, ops

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from modules.autoencoder import SD_CONFIG, AutoencoderKL

from examples.dit.pipelines.infer_pipeline import DiTInferPipeline
from mindone.models.dit import VideoDiT_models
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.ms_mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.ms_mode,
        device_target=args.device_target,
        device_id=device_id,
    )

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
        help="image size in [256, 512]",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="number of frames",
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
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    parser.add_argument(
        "--use_fp16",
        default=True,
        type=str2bool,
        help="whether to use fp16 for DiT mode. Default is True",
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
    dit_model = VideoDiT_models[args.model_name](
        input_size=latent_size,
        num_classes=1000,
        block_kwargs={"enable_flash_attention": args.enable_flash_attention},
    )
    amp_level = "O2" if args.use_fp16 else "O1"
    dit_model = auto_mixed_precision(dit_model, amp_level=amp_level)
    dit_model.load_params_from_dit_ckpt(args.dit_checkpoint)
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
    class_labels = [207]
    # Create sampling noise:
    n = len(class_labels)
    z = ops.randn((n, args.num_frames, 4, latent_size, latent_size), dtype=ms.float32)
    y = Tensor(class_labels)
    y_null = ops.ones_like(y) * 1000

    # 3. build inference pipeline
    pipeline = DiTInferPipeline(
        dit_model,
        vae,
        scale_factor=args.sd_scale_factor,
        num_inference_steps=args.sampling_steps,
        guidance_rescale=args.guidance_scale,
    )

    # 4. print key info
    num_params_vae, num_params_vae_trainable = count_params(vae)
    num_params_dit, num_params_dit_trainable = count_params(dit_model)
    num_params = num_params_vae + num_params_dit
    num_params_trainable = num_params_vae_trainable + num_params_dit_trainable
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
            f"Class Labels: {class_labels}",
            f"Num params: {num_params:,} (dit: {num_params_dit:,}, vae: {num_params_vae:,})",
            f"Num trainable params: {num_params_trainable:,}",
            f"AMP Level: {amp_level}",
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
    for i, class_label in enumerate(class_labels, 0):
        save_fp = f"{save_dir}/class-{class_label}.gif"
        save_videos(x_samples, save_fp, loop=0)
        logger.info(f"save to {save_fp}")
