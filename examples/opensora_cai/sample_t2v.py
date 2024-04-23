import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

import mindspore as ms
from mindspore import nn

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from opensora.models.autoencoder import SD_CONFIG, AutoencoderKL
from opensora.models.layers.blocks import Attention, LayerNorm
from opensora.models.stdit import STDiT_XL_2
from opensora.models.text_encoders import get_text_encoder_and_tokenizer
from opensora.pipelines import InferPipeline
from opensora.utils.model_utils import _check_cfgs_in_parser, count_params, str2bool

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)


def init_env(args):
    ms.set_context(mode=args.mode)
    ms.set_context(
        mode=args.mode,
        device_target=args.device_target,
    )


def read_captions_from_csv(csv_path, caption_column="caption"):
    df = pd.read_csv(csv_path, usecols=[caption_column])
    captions = df[caption_column].values.tolist()
    return captions


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    init_env(args)
    set_random_seed(args.seed)

    # get captions from cfg or prompt_file
    if args.prompt_file is not None:
        if args.prompt_file.endswith(".csv"):
            captions = read_captions_from_csv(args.prompt_file)
        elif args.prompt_file.endswith(".txt"):
            captions = []
            with open(args.caption_file, "r") as fp:
                for line in fp:
                    captions.append(line.strip())
    else:
        captions = args.captions

    # 2. model initiate and weight loading
    # 2.1 latte
    logger.info("STDiT init")

    VAE_T_COMPRESS = 1
    VAE_S_COMPRESS = 8
    VAE_Z_CH = SD_CONFIG["z_channels"]
    input_size = (
        args.num_frames // VAE_T_COMPRESS,
        args.image_size // VAE_S_COMPRESS,
        args.image_size // VAE_S_COMPRESS,
    )
    if args.image_size == 512 and args.space_scale == 0.5:
        logger.warning("space_ratio should be 1 for 512x512 resolution")
    model_extra_args = dict(
        input_size=input_size,
        in_channels=VAE_Z_CH,
        space_scale=args.space_scale,  # 0.5 for 256x256. 1. for 512
        time_scale=args.time_scale,
        patchify_conv3d_replace="conv2d",  # for Ascend
        enable_flashattn=args.enable_flash_attention,
    )
    latte_model = STDiT_XL_2(**model_extra_args)
    latte_model = latte_model.set_train(False)

    if args.dtype == "fp32":
        model_dtype = ms.float32
    else:
        model_dtype = {"fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]
        latte_model = auto_mixed_precision(
            latte_model,
            amp_level="O2",
            dtype=model_dtype,
            custom_fp32_cells=[LayerNorm, Attention, nn.SiLU, nn.GELU], # NOTE: keep it the same as training setting
        )

    if len(args.checkpoint) > 0:
        logger.info(f"Loading ckpt {args.checkpoint} into STDiT")
        latte_model.load_from_checkpoint(args.checkpoint)
    else:
        logger.warning("STDiT uses random initialization!")

    # 2.2 vae
    logger.info("vae init")
    vae = AutoencoderKL(
        SD_CONFIG,
        VAE_Z_CH,
        ckpt_path=args.vae_checkpoint,
        use_fp16=False,
    )
    vae = vae.set_train(False)

    # 2.3 text encoder
    if args.embed_path is None:
        text_encoder, tokenizer = get_text_encoder_and_tokenizer("t5", args.t5_model_dir)
        n = len(captions)
        text_tokens, mask = text_encoder.get_text_tokens_and_mask(captions, return_tensor=True)
        text_emb = None
    else:
        dat = np.load(args.embed_path)
        text_tokens, mask, text_emb = dat["tokens"], dat["mask"], dat["text_emb"]
        n = text_emb.shape[0]
        text_tokens = ms.Tensor(text_tokens)
        mask = ms.Tensor(mask, dtype=ms.uint8)
        text_emb = ms.Tensor(text_emb, dtype=ms.float32)
        text_encoder = None
    assert n > 0, "No captions provided"
    logger.info(f"Num tokens: {mask.asnumpy().sum(1)}")

    # 3. build inference pipeline
    pipeline = InferPipeline(
        latte_model,
        vae,
        text_encoder=text_encoder,
        scale_factor=args.sd_scale_factor,
        num_inference_steps=args.sampling_steps,
        guidance_rescale=args.guidance_scale,
        ddim_sampling=args.ddim_sampling,
        condition="text",
    )

    # 4. print key info
    num_params_vae, num_params_vae_trainable = count_params(vae)
    num_params_latte, num_params_latte_trainable = count_params(latte_model)
    num_params = num_params_vae + num_params_latte
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Num of samples: {n}",
            f"Num params: {num_params:,} (latte: {num_params_latte:,}, vae: {num_params_vae:,})",
            f"Use model dtype: {model_dtype}",
            f"Sampling steps {args.sampling_steps}",
            f"DDIM sampling: {args.ddim_sampling}",
            f"CFG guidance scale: {args.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    for i in range(0, len(captions), args.batch_size):
        batch_prompts = captions[i : i + args.batch_size]
        ns = len(batch_prompts)

        # prepare inputs
        inputs = {}
        # b c t h w
        # z = ops.randn([ns, VAE_Z_CH] + list(input_size), dtype=ms.float32)
        z = np.random.randn(*([ns, VAE_Z_CH] + list(input_size)))  # for ensure generate the same noise
        z = ms.Tensor(z, dtype=ms.float32)
        inputs["noise"] = z
        inputs["scale"] = args.guidance_scale
        if text_emb is None:
            inputs["text_tokens"] = text_tokens[i : i + ns]
            inputs["text_emb"] = None
            inputs["mask"] = mask[i : i + ns]
        else:
            inputs["text_tokens"] = None
            inputs["text_emb"] = text_emb[i : i + ns]
            inputs["mask"] = mask[i : i + ns]

        logger.info("Sampling for captions: ")
        for j in range(ns):
            logger.info(captions[i + j])

        # infer
        start_time = time.time()
        x_samples = pipeline(inputs, latent_save_fp=f"outputs/denoised_latent_{i:02d}.npy")
        x_samples = x_samples.asnumpy()
        batch_time = time.time() - start_time

        logger.info(
            f"Batch time cost: {batch_time:.3f}s, sampling speed: {args.sampling_steps*ns/batch_time:.2f} step/s"
        )

        # save result
        for j in range(ns):
            global_idx = i * args.batch_size + j
            prompt = "-".join((batch_prompts[j].replace("/", "").split(" ")[:10]))
            save_fp = f"{save_dir}/{global_idx:03d}-{prompt}.{args.save_format}"
            save_videos(x_samples[j : j + 1], save_fp, fps=args.fps)
            logger.info(f"save to {save_fp}")


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
        "--num_samples",
        type=int,
        default=3,
        help="number of videos to be generated unconditionally. If using text or class as conditions,"
        " the number of samples will be defined by the number of class labels or text captions",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="latte checkpoint path. If specified, will load from it, otherwise, will use random initialization",
    )
    parser.add_argument("--t5_model_dir", default=None, type=str, help="the T5 cache folder path")
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
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument("--space_scale", default=0.5, type=float, help="stdit model space scalec")
    parser.add_argument("--time_scale", default=1.0, type=float, help="stdit model time scalec")
    parser.add_argument(
        "--captions",
        type=str,
        nargs="+",
        help="A list of text captions to be generated with",
    )
    parser.add_argument("--prompt_file", default=None, type=str, help="path to a csv file containing captions")
    parser.add_argument(
        "--save_format",
        default="mp4",
        choices=["gif", "mp4"],
        type=str,
        help="video format for saving the sampling output, gif or mp4",
    )
    parser.add_argument("--fps", type=int, default=8, help="FPS in the saved video")
    parser.add_argument("--batch_size", default=4, type=int, help="infer batch size")
    parser.add_argument("--embed_path", type=str, default=None, help="path to t5 embedding")
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
    args = parse_args()
    main(args)
