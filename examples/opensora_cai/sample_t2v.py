import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np
import yaml

import mindspore as ms
from mindspore import Tensor, ops

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from opensora.models.autoencoder import SD_CONFIG, AutoencoderKL
from opensora.models.stdit import STDiT_XL_2
from opensora.models.text_encoders import get_text_encoder_and_tokenizer
from opensora.pipelines import InferPipeline
from opensora.utils.model_utils import _check_cfgs_in_parser, count_params, remove_pname_prefix, str2bool

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.mode)
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.mode,
        device_target=args.device_target,
        device_id=device_id,
    )
    if args.precision_mode is not None:
        ms.set_context(ascend_config={"precision_mode": args.precision_mode})
    return device_id


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    init_env(args)
    set_random_seed(args.seed)

    # 2. model initiate and weight loading
    # 2.1 latte
    logger.info(f"{args.model_name}-{args.image_size}x{args.image_size} init")

    vae_t_compress = 1
    vae_s_compress = 8
    vae_out_channels = 4

    text_emb_dim = 4096
    max_tokens = 120

    if args.image_size == 256:
        space_scale = 0.5
    elif args.image_size == 512:
        space_scale = 1.0
    else:
        raise ValueError

    input_size = (
        args.num_frames // vae_t_compress,
        args.image_size // vae_s_compress,
        args.image_size // vae_s_compress,
    )

    # FIXME: set this parameter by config file
    model_extra_args = dict(
        input_size=input_size,
        in_channels=vae_out_channels,
        caption_channels=text_emb_dim,
        model_max_length=max_tokens,
        space_scale=space_scale,  # 0.5 for 256x256. 1. for 512. # TODO: align to torch
        time_scale=1.0,
        patchify_conv3d_replace="conv2d",  # for Ascend
    )
    latte_model = STDiT_XL_2(**model_extra_args)

    if args.dtype == "fp16":
        model_dtype = ms.float16
        latte_model = auto_mixed_precision(latte_model, amp_level="O2", dtype=model_dtype)
    elif args.dtype == "bf16":
        model_dtype = ms.bfloat16
        latte_model = auto_mixed_precision(latte_model, amp_level="O2", dtype=model_dtype)
    else:
        model_dtype = ms.float32

    if len(args.checkpoint) > 0:
        logger.info(f"Loading ckpt {args.checkpoint} into STDiT")
        latte_model.load_from_checkpoint(args.checkpoint)
    else:
        logger.warning("STDiT uses random initialization!")

    latte_model = latte_model.set_train(False)
    for param in latte_model.get_parameters():  # freeze latte_model
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

    if args.text_encoder == "t5":
        ckpt_path = args.t5_model_dir
    elif args.text_encoder == "clip":
        ckpt_path = args.clip_checkpoint

    if args.embed_path is None:
        text_encoder, tokenizer = get_text_encoder_and_tokenizer(args.text_encoder, ckpt_path)
        n = len(args.captions)
        assert n > 0, "No captions provided"
        text_tokens, mask = text_encoder.get_text_tokens_and_mask(args.captions, return_tensor=True)
        text_emb = None
    else:
        dat = np.load(args.embed_path)
        text_tokens, mask, text_emb = dat["tokens"], dat["mask"], dat["text_emb"]
        n = text_emb.shape[0]
        text_tokens = ms.Tensor(text_tokens)
        mask = ms.Tensor(mask, dtype=ms.uint8)
        text_emb = ms.Tensor(text_emb, dtype=ms.float32)
        text_encoder = None
        tokenizer = None

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
        condition=args.condition,
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

    for i in range(0, len(args.captions), args.batch_size):
        batch_prompts = args.captions[i : i + arg.batch_size]
        ns = len(batch_prompts)

        # prepare inputs
        inputs = {}
        # b c t h w
        # z = ops.randn([ns, vae_out_channels] + list(input_size), dtype=ms.float32)
        z = np.random.randn(*([ns, vae_out_channels] + list(input_size))) # for ensure generate the same noise
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

        logger.info(f"Sampling for {n} samples with captions: ")
        for i in range(n):
            logger.info(args.captions[i])

        start_time = time.time()

        # infer
        x_samples = pipeline(inputs, latent_save_fp=f"outputs/denoised_latent_{i:02d}.npy")
        x_samples = x_samples.asnumpy()

        end_time = time.time()

        # save result
        for j in range(args.batch_size):
            save_fp = f"{save_dir}/{i:02d}-{j:02d}.gif"
            save_videos(x_samples[j : j + 1], save_fp, loop=0)
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
        "--model_name",
        "-m",
        type=str,
        default="STDiT-XL/2",
        help="Model name ",
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
    parser.add_argument("--t5_model_dir", default=None, type=str, help="the T5 cache folder path")
    parser.add_argument(
        "--clip_checkpoint",
        type=str,
        default=None,
        help="CLIP text encoder checkpoint (or sd checkpoint to only load the text encoder part.)",
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
    parser.add_argument("--batch_size", default=2, type=int, help="infer batch size")
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
