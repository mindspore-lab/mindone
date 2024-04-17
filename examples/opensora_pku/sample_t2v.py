import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np
import yaml
from omegaconf import OmegaConf
from utils.model_utils import _check_cfgs_in_parser, remove_pname_prefix

import mindspore as ms

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from opensora.models.diffusion.latte_t2v import LatteT2V
from opensora.sample.pipeline_videogen import VideoGenPipeline
from opensora.text_encoders.t5_embedder import T5Embedder

from mindone.diffusers.schedulers import DDIMScheduler, DDPMScheduler
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import instantiate_from_config, str2bool
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

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
        "--vae_config",
        type=str,
        default="configs/ae/causal_vae_488.yaml",
        help="path to load a config yaml file that describes the VAE model",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="image size in [256, 512]",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="number of classes, applies only when condition is `class`",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="number of videos to be generated unconditionally. If using text or class as conditions,"
        " the number of samples will be defined by the number of class labels or text captions",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="17x256x256",
        help="Model version in ['17x256x256', '65x256x256', '65x512x512'] ",
    )
    parser.add_argument(
        "--condition",
        default=None,
        type=str,
        help="the condition types: `None` means using no conditions; `text` means using text embedding as conditions;"
        " `class` means using class labels as conditions.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="latte checkpoint path. If specified, will load from it, otherwise, will use random initialization",
    )
    parser.add_argument(
        "--text_encoder",
        default=None,
        type=str,
        choices=["clip", "t5"],
        help="text encoder for extract text embeddings: clip text encoder or t5-v1_1-xxl.",
    )
    parser.add_argument("--t5_cache_folder", default=None, type=str, help="the T5 cache folder path")
    parser.add_argument(
        "--clip_checkpoint",
        type=str,
        default=None,
        help="CLIP text encoder checkpoint (or sd checkpoint to only load the text encoder part.)",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/ae/causal_vae_488.ckpt",
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
        "--use_recompute",
        default=False,
        type=str2bool,
        help="whether use recompute.",
    )
    parser.add_argument(
        "--patch_embedder",
        type=str,
        default="conv",
        choices=["conv", "linear"],
        help="Whether to use conv2d layer or dense (linear layer) as Patch Embedder.",
    )
    parser.add_argument(
        "--captions",
        type=str,
        nargs="+",
        help="A list of text captions to be generated with",
    )
    parser.add_argument(
        "--num_videos_per_prompt", type=int, default=1, help="the number of images to be generated for each prompt"
    )
    parser.add_argument("--ddim_sampling", type=str2bool, default=True, help="Whether to use DDIM for sampling")
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
    # 2.1 latte
    logger.info(f"Latte-{args.model_version} init")

    assert args.condition == "text", "LatteT2V only support text condition now!"
    assert args.text_encoder == "t5", "LatteT2V only support t5 text encoder now!"

    latte_model = LatteT2V.from_pretrained_2d(
        "models",
        subfolder=args.model_version,
        enable_flash_attention=args.enable_flash_attention,
        use_recompute=args.use_recompute,
    )
    if args.dtype == "fp16":
        model_dtype = ms.float16
        latte_model = auto_mixed_precision(latte_model, amp_level="O2", dtype=model_dtype)
    elif args.dtype == "bf16":
        raise ValueError("LatteT2V only support fp16 and fp32 now!")
    else:
        model_dtype = ms.float32
        latte_model = latte_model.to(model_dtype)
    video_length, image_size = latte_model.config.video_length, args.image_size
    # latent_size = (image_size // ae_stride_config[args.ae][1], image_size // ae_stride_config[args.ae][2])
    latent_size = args.image_size // 8

    if len(args.checkpoint) > 0:
        param_dict = ms.load_checkpoint(args.checkpoint)
        logger.info(f"Loading ckpt {args.checkpoint} into Latte")
        # in case a save ckpt with "network." prefix, removing it before loading
        param_dict = remove_pname_prefix(param_dict, prefix="network.")
        latte_model.load_params_from_ckpt(param_dict)
    else:
        logger.warning("Latte uses random initialization!")

    latte_model = latte_model.set_train(False)
    for param in latte_model.get_parameters():  # freeze latte_model
        param.requires_grad = False

    # 2.2 vae
    logger.info("vae init")
    config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(config.generator)
    vae.init_from_ckpt(args.vae_checkpoint)
    vae.set_train(False)

    vae = auto_mixed_precision(vae, amp_level="O2", dtype=ms.float16)
    logger.info("Use amp level O2 for causal 3D VAE.")

    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False
    vae.latent_size = (latent_size, latent_size)

    assert args.condition == "text", "LatteT2V only support text condition"
    assert args.text_encoder == "t5", "LatteT2V only support t5 text encoder"
    logger.info("T5 init")
    text_encoder = T5Embedder(
        cache_dir=args.t5_cache_folder, pretrained_ckpt=os.path.join(args.t5_cache_folder, "model.ckpt")
    )
    tokenizer = text_encoder.tokenizer
    n = len(args.captions)
    assert n > 0, "No captions provided"

    # 3. build inference pipeline
    scheduler = DDIMScheduler() if args.ddim_sampling else DDPMScheduler()
    text_encoder = text_encoder.model
    pipeline = VideoGenPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=latte_model,
        vae_scale_factor=args.sd_scale_factor,
    )

    # 4. print key info
    num_params_vae, num_params_vae_trainable = count_params(vae)
    num_params_latte, num_params_latte_trainable = count_params(latte_model)
    num_params = num_params_vae + num_params_latte
    num_params_trainable = num_params_vae_trainable + num_params_latte_trainable
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Num of samples: {n}",
            f"Num params: {num_params:,} (latte: {num_params_latte:,}, vae: {num_params_vae:,})",
            f"Num trainable params: {num_params_trainable:,}",
            f"Use model dtype: {model_dtype}",
            f"Sampling steps {args.sampling_steps}",
            f"DDIM sampling: {args.ddim_sampling}",
            f"CFG guidance scale: {args.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    logger.info(f"Sampling for {n} samples with condition {args.condition}")
    start_time = time.time()

    # infer
    video_grids = []
    if not isinstance(args.captions, list):
        args.captions = [args.captions]
    if len(args.captions) == 1 and args.captions[0].endswith("txt"):
        captions = open(args.captions[0], "r").readlines()
        args.captions = [i.strip() for i in captions]
    for prompt in args.captions:
        print("Processing the ({}) prompt".format(prompt))
        videos = pipeline(
            prompt,
            video_length=video_length,
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.sampling_steps,
            guidance_scale=args.guidance_scale,
            enable_temporal_attentions=True,
            num_videos_per_prompt=args.num_videos_per_prompt,
            mask_feature=False,
        ).video.asnumpy()
        video_grids.append(videos)
    x_samples = np.stack(video_grids, axis=0)

    end_time = time.time()

    # save result
    for i in range(n):
        for i_video in range(args.num_videos_per_prompt):
            save_fp = f"{save_dir}/{i_video}-{args.captions[i].strip()[:100]}.gif"
            save_video_data = x_samples[i : i + 1, i_video].transpose(0, 2, 3, 4, 1)  # (b c t h w) -> (b t h w c)
            save_videos(save_video_data, save_fp, loop=0)
            logger.info(f"save to {save_fp}")
