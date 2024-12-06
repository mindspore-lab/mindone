import argparse
import datetime
import glob
import logging
import os
import sys
import time

import numpy as np
import yaml

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.communication.management import GlobalComm, get_group_size, get_rank, init

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets.aspect import ASPECT_RATIO_MAP, ASPECT_RATIOS, get_image_size, get_num_frames
from opensora.models.stdit import STDiT2_XL_2, STDiT3_XL_2, STDiT3_XL_2_DSP, STDiT_XL_2
from opensora.models.text_encoder.t5 import get_text_encoder_and_tokenizer
from opensora.models.vae.vae import SD_CONFIG, OpenSoraVAE_V1_2, VideoAutoencoderKL
from opensora.pipelines import InferPipeline, InferPipelineFiTLike
from opensora.utils.amp import auto_mixed_precision
from opensora.utils.cond_data import get_references, read_captions_from_csv, read_captions_from_txt
from opensora.utils.model_utils import WHITELIST_OPS, _check_cfgs_in_parser, str2bool
from opensora.utils.util import IMG_FPS, apply_mask_strategy, process_mask_strategies, process_prompts

from mindone.data.data_split import distribute_samples
from mindone.utils.logger import set_logger
from mindone.utils.misc import to_abspath
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)


def to_numpy(x: Tensor) -> np.ndarray:
    if x.dtype == ms.bfloat16:
        x = x.astype(ms.float32)
    return x.asnumpy()


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    jit_level: str = "O0",
    enable_sequence_parallelism: bool = False,
    debug: bool = False,
):
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)
    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if debug and mode == ms.GRAPH_MODE:  # force PyNative mode when debugging
        logger.warning("Debug mode is on, switching execution mode to PyNative.")
        mode = ms.PYNATIVE_MODE

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
        ms.reset_auto_parallel_context()

        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )

        if enable_sequence_parallelism:
            set_sequence_parallel_group(GlobalComm.WORLD_COMM_GROUP)
            ms.set_auto_parallel_context(enable_alltoall=True)
    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
            pynative_synchronize=debug,
        )

        if enable_sequence_parallelism:
            raise ValueError("`use_parallel` must be `True` when using sequence parallelism.")

    try:
        if jit_level in ["O0", "O1", "O2"]:
            ms.set_context(jit_config={"jit_level": jit_level})
        else:
            logger.warning(
                f"Unsupport jit_level: {jit_level}. The framework automatically selects the execution method"
            )
    except Exception:
        logger.warning(
            "The current jit_level is not suitable because current MindSpore version or mode does not match,"
            "please ensure the MindSpore version >= ms2.3_0615, and use GRAPH_MODE."
        )

    return rank_id, device_num


def main(args):
    if args.append_timestr:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir = f"{args.output_path}/{time_str}"
    else:
        save_dir = f"{args.output_path}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    latent_dir = os.path.join(save_dir, "denoised_latents")
    if args.save_latent:
        os.makedirs(latent_dir, exist_ok=True)

    # 1. init env
    rank_id, device_num = init_env(
        args.mode,
        args.seed,
        args.use_parallel,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
        jit_level=args.jit_level,
        debug=args.debug,
        enable_sequence_parallelism=args.enable_sequence_parallelism,
    )

    # 1.1 get captions from cfg or prompt_path
    if args.prompt_path is not None:
        if args.prompt_path.endswith(".csv"):
            captions = read_captions_from_csv(args.prompt_path)
        else:  # treat any other file as a plain text
            captions = read_captions_from_txt(args.prompt_path)
    else:
        captions = args.captions

    align, latent_condition_frame_length = 0, args.condition_frame_length  # frames alignment for video looping
    if args.model_version == "v1" and args.loop > 1:
        args.loop = 1
        logger.warning("OpenSora v1 doesn't support iterative video generation. Setting loop to 1.")
    elif args.model_version == "v1.2":
        align = 5
        latent_condition_frame_length = round(latent_condition_frame_length / 17 * 5)

    captions = process_prompts(captions, args.loop)  # in v1.1 and above, each loop can have a different caption
    if not args.enable_sequence_parallelism:
        # split samples to NPUs as even as possible
        start_idx, end_idx = distribute_samples(len(captions), rank_id, device_num)
        captions = captions[start_idx:end_idx]
        base_data_idx = start_idx
    else:
        base_data_idx = 0

    if args.use_parallel and not args.enable_sequence_parallelism:
        print(f"Num captions for rank {rank_id}: {len(captions)}")

    # 2. model initiate and weight loading
    # 2.1 vae
    dtype_map = {"fp32": ms.float32, "fp16": ms.float16, "bf16": ms.bfloat16}
    # if args.use_vae_decode or args.reference_path is not None:
    # TODO: fix vae get_latent_size for vae cache
    logger.info("vae init")
    if args.vae_type in [None, "VideoAutoencoderKL"]:
        # vae = AutoencoderKL(SD_CONFIG, VAE_Z_CH, ckpt_path=args.vae_checkpoint)
        vae = VideoAutoencoderKL(
            config=SD_CONFIG, ckpt_path=args.vae_checkpoint, micro_batch_size=args.vae_micro_batch_size
        )
    elif args.vae_type == "OpenSoraVAE_V1_2":
        assert os.path.exists(args.vae_checkpoint), f"vae checkopint {args.vae_checkpoint} NOT found"
        vae = OpenSoraVAE_V1_2(
            micro_batch_size=args.vae_micro_batch_size,
            micro_frame_size=args.vae_micro_frame_size,
            ckpt_path=args.vae_checkpoint,
            freeze_vae_2d=True,
        )
    else:
        raise ValueError(f"Unknown VAE type: {args.vae_type}")

    vae = vae.set_train(False)
    if args.vae_dtype in ["fp16", "bf16"]:
        vae = auto_mixed_precision(
            vae, amp_level=args.amp_level, dtype=dtype_map[args.vae_dtype], custom_fp32_cells=[nn.GroupNorm]
        )

    VAE_Z_CH = vae.out_channels
    if args.image_size is not None:
        img_h, img_w = args.image_size if isinstance(args.image_size, list) else (args.image_size, args.image_size)
    else:
        if args.resolution is None or args.aspect_ratio is None:
            raise ValueError("`resolution` and `aspect_ratio` must be provided if `image_size` is not provided")
        img_h, img_w = get_image_size(args.resolution, args.aspect_ratio)

    num_frames = get_num_frames(args.num_frames)
    latent_size = vae.get_latent_size((num_frames, img_h, img_w))

    # 2.2 latte
    patchify_conv3d_replace = "linear" if args.pre_patchify else args.patchify
    model_extra_args = dict(
        input_size=latent_size,
        in_channels=VAE_Z_CH,
        model_max_length=args.model_max_length,
        patchify_conv3d_replace=patchify_conv3d_replace,  # for Ascend
        enable_flashattn=args.enable_flash_attention,
        enable_sequence_parallelism=args.enable_sequence_parallelism,
    )
    if args.pre_patchify and args.model_version != "v1.1":
        raise ValueError("`pre_patchify=True` can only be used in model version 1.1.")

    if args.model_version == "v1":
        model_name = "STDiT"
        if img_h != img_w:
            raise ValueError(f"OpenSora v1 support square images only, but got {args.image_size}")

        if args.image_size == 512 and args.space_scale != 1:
            logger.warning("space_ratio should be 1 for 512x512 resolution")

        model_extra_args.update(
            {
                "space_scale": args.space_scale,  # 0.5 for 256x256. 1. for 512
                "time_scale": args.time_scale,
            }
        )

        logger.info(f"{model_name} init")
        latte_model = STDiT_XL_2(**model_extra_args)

    elif args.model_version == "v1.1":
        model_name = "STDiT2"
        model_extra_args.update({"input_sq_size": 512, "qk_norm": True})
        logger.info(f"{model_name} init")
        latte_model = STDiT2_XL_2(**model_extra_args)
    elif args.model_version == "v1.2":
        model_name = "STDiT3"
        model_extra_args["qk_norm"] = True
        logger.info(f"{model_name} init")
        if args.dsp:
            latte_model = STDiT3_XL_2_DSP(**model_extra_args)
        else:
            latte_model = STDiT3_XL_2(**model_extra_args)
    else:
        raise ValueError(f"Unknown model version: {args.model_version}")

    latte_model = latte_model.set_train(False)

    if latent_size[1] % latte_model.patch_size[1] != 0 or latent_size[2] % latte_model.patch_size[2] != 0:
        height_ = latte_model.patch_size[1] * 8
        width_ = latte_model.patch_size[2] * 8
        msg = f"Image height ({img_h}) and width ({img_w}) should be divisible by {height_} and {width_} respectively."
        if patchify_conv3d_replace == "linear":
            raise ValueError(msg)
        else:
            logger.warning(msg)

    if args.dtype in ["fp16", "bf16"]:
        latte_model = auto_mixed_precision(
            latte_model, amp_level=args.amp_level, dtype=dtype_map[args.dtype], custom_fp32_cells=WHITELIST_OPS
        )

    if args.ckpt_path:
        logger.info(f"Loading ckpt {args.ckpt_path} into {model_name}")
        assert os.path.exists(args.ckpt_path), f"{args.ckpt_path} not found."
        latte_model.load_from_checkpoint(args.ckpt_path)
    else:
        logger.warning(f"{model_name} uses random initialization!")

    # 2.3 text encoder
    if args.text_embed_folder is None:
        text_encoder, tokenizer = get_text_encoder_and_tokenizer(
            "t5", args.t5_model_dir, model_max_length=args.model_max_length
        )
        num_prompts = len(captions)
        text_tokens, mask = zip(
            *[text_encoder.get_text_tokens_and_mask(caption, return_tensor=False) for caption in captions]
        )
        text_tokens, mask = Tensor(text_tokens, dtype=ms.int32), Tensor(mask, dtype=ms.uint8)
        text_emb = None
        # TODO: use FA in T5
        if args.t5_dtype in ["fp16", "bf16"]:
            if args.t5_dtype == "fp16":
                logger.warning(
                    "T5 dtype is fp16, which may lead to video color vibration. Suggest to use bf16 or fp32."
                )
            text_encoder = auto_mixed_precision(
                text_encoder, amp_level="O2", dtype=dtype_map[args.t5_dtype], custom_fp32_cells=WHITELIST_OPS
            )
        logger.info(f"Num tokens: {mask.asnumpy().sum(2)}")
    else:
        assert not args.use_parallel, "parallel inference is not supported for t5 cached sampling currently."
        if args.model_version != "v1":
            logger.warning("For embedded captions, only one prompt per video is supported at this moment.")

        embed_paths = sorted(glob.glob(os.path.join(args.text_embed_folder, "*.npz")))
        prompt_prefix = []
        text_tokens, mask, text_emb = [], [], []
        for fp in embed_paths:
            prompt_prefix.append(os.path.basename(fp)[:-4])
            dat = np.load(fp)
            text_tokens.append(dat["tokens"])
            mask.append(dat["mask"])
            text_emb.append(dat["text_emb"])
        text_tokens = np.concatenate(text_tokens)
        mask = np.concatenate(mask)
        text_emb = np.concatenate(text_emb)
        logger.info(f"Num tokens: {mask.sum(1)}")

        num_prompts = text_emb.shape[0]
        text_tokens = ms.Tensor(text_tokens)
        mask = ms.Tensor(mask, dtype=ms.uint8)
        text_emb = ms.Tensor(text_emb, dtype=ms.float32)
        text_encoder = None

    if (args.model_version == "v1" or args.reference_path is None) and num_prompts < 1:
        raise ValueError("No text prompts provided for Text-to-Video generation.")

    # 3. build inference pipeline
    pipeline_kwargs = dict(
        scale_factor=args.sd_scale_factor,
        num_inference_steps=args.sampling_steps,
        guidance_rescale=args.guidance_scale,
        guidance_channels=args.guidance_channels,
        sampling=args.sampling,
        micro_batch_size=args.vae_micro_batch_size,
    )
    if args.pre_patchify:
        additional_pipeline_kwargs = dict(
            patch_size=latte_model.patch_size,
            max_image_size=args.max_image_size,
            max_num_frames=args.max_num_frames,
            embed_dim=latte_model.hidden_size,
            num_heads=latte_model.num_heads,
            vae_downsample_rate=8.0,
            in_channels=latte_model.in_channels,
            input_sq_size=latte_model.input_sq_size,
        )
        pipeline_kwargs.update(additional_pipeline_kwargs)

    # TODO: need to adapt new vae to FiT
    pipeline_ = InferPipelineFiTLike if args.pre_patchify else InferPipeline
    pipeline = pipeline_(latte_model, vae, text_encoder=text_encoder, **pipeline_kwargs)

    # 3.1. Support for multi-resolution (OpenSora v1.1 and above)
    model_args = {}
    if args.model_version != "v1":
        model_args["height"] = Tensor([img_h] * args.batch_size, dtype=dtype_map[args.dtype])
        model_args["width"] = Tensor([img_w] * args.batch_size, dtype=dtype_map[args.dtype])
        model_args["num_frames"] = Tensor([num_frames] * args.batch_size, dtype=dtype_map[args.dtype])
        model_args["ar"] = Tensor([img_h / img_w] * args.batch_size, dtype=dtype_map[args.dtype])
        fps = args.fps if num_frames > 1 else IMG_FPS
        model_args["fps"] = Tensor([fps] * args.batch_size, dtype=dtype_map[args.dtype])

    # 3.2 Prepare references (OpenSora v1.1 and above)
    if args.reference_path is not None and not (len(args.reference_path) == 1 and args.reference_path[0] == ""):
        if len(args.reference_path) != num_prompts:
            raise ValueError(f"Reference path mismatch: {len(args.reference_path)} != {num_prompts}")
        if len(args.reference_path) != len(args.mask_strategy):
            raise ValueError(f"Mask strategy mismatch: {len(args.mask_strategy)} != {len(captions)}")
    else:
        args.reference_path = [None] * len(captions)
        args.mask_strategy = [None] * len(captions)
    frames_mask_strategies = process_mask_strategies(args.mask_strategy)

    # 4. print key info
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Num of captions: {num_prompts}",
            f"dtype: {args.dtype}",
            f"amp_level: {args.amp_level}",
            f"Image size: {(img_h, img_w)}",
            f"Num frames: {num_frames}",
            f"Sampling steps {args.sampling_steps}",
            f"Sampling: {args.sampling}",
            f"CFG guidance scale: {args.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    for i in range(0, num_prompts, args.batch_size):
        if text_emb is None:
            batch_prompts = captions[i : i + args.batch_size]
            ns = len(batch_prompts)
        else:
            ns = min(args.batch_size, text_emb.shape[0] - i)

        frames_mask_strategy = frames_mask_strategies[i : i + args.batch_size]

        references = get_references(args.reference_path[i : i + args.batch_size], (img_h, img_w))
        # embed references into latent space
        for ref in references:
            if ref is not None:
                for k in range(len(ref)):
                    try:
                        ref[k] = pipeline.vae_encode(Tensor(ref[k])).asnumpy()[0]
                    except RuntimeError as e:
                        logger.error(
                            f"Failed to embed reference video {args.reference_path[i : i + args.batch_size][k]}."
                            f" Try reducing `vae_micro_batch_size`."
                        )
                        raise e

        frames_mask, latents, videos = None, [], []
        for loop_i in range(args.loop):
            if loop_i > 0:
                for j in range(len(references)):  # iterate over batch of references
                    if references[j] is None:
                        references[j] = [latents[-1][j]]
                    else:
                        references[j].append(latents[-1][j])
                    new_strategy = [
                        loop_i,
                        len(references[j]) - 1,
                        -latent_condition_frame_length,
                        0,
                        latent_condition_frame_length,
                        args.condition_frame_edit,
                    ]
                    if frames_mask_strategy[j] is None:
                        frames_mask_strategy[j] = [new_strategy]
                    else:
                        frames_mask_strategy[j].append(new_strategy)

            # prepare inputs
            inputs = {}
            # b c t h w
            z = np.random.randn(ns, VAE_Z_CH, *latent_size).astype(np.float32)

            if args.model_version != "v1":
                z, frames_mask = apply_mask_strategy(z, references, frames_mask_strategy, loop_i, align)
                frames_mask = Tensor(frames_mask, dtype=ms.float32)

            z = ms.Tensor(z, dtype=ms.float32)
            inputs["noise"] = z
            inputs["scale"] = args.guidance_scale
            if text_emb is None:
                inputs["text_tokens"] = text_tokens[i : i + ns, loop_i]
                inputs["text_emb"] = None
                inputs["mask"] = mask[i : i + ns, loop_i]
            else:
                inputs["text_tokens"] = None
                inputs["text_emb"] = text_emb[i : i + ns]
                inputs["mask"] = mask[i : i + ns]

            logger.info("Sampling captions:")
            for j in range(ns):
                if text_emb is None:
                    logger.info(captions[i + j][loop_i])
                else:
                    logger.info(prompt_prefix[i + j])

            # infer
            start_time = time.time()
            samples, latent = pipeline(
                inputs, frames_mask=frames_mask, num_frames=num_frames, additional_kwargs=model_args
            )
            # TODO: adjust to decoder time compression
            latents.append(to_numpy(latent)[:, :, latent_condition_frame_length if loop_i > 0 else 0 :])
            if samples is not None:
                videos.append(to_numpy(samples)[:, args.condition_frame_length if loop_i > 0 else 0 :])
            batch_time = time.time() - start_time
            logger.info(
                f"Batch time cost: {batch_time:.3f}s, sampling speed: {args.sampling_steps * ns / batch_time:.4f} step/s"
            )

        latents = np.concatenate(latents, axis=2)
        if videos:
            videos = np.concatenate(videos, axis=1)

        if args.enable_sequence_parallelism and rank_id > 0:
            # no need to save images for rank > 1
            continue

        # save result
        for j in range(ns):
            global_idx = base_data_idx + i + j
            if args.text_embed_folder is None:
                prompt = "-".join((batch_prompts[j][0].replace("/", "").split(" ")[:10]))
                save_fp = f"{save_dir}/{global_idx:03d}-{prompt}.{args.save_format}"
                latent_save_fp = f"{latent_dir}/{global_idx:03d}-{prompt}.npy"
            else:
                fn = prompt_prefix[global_idx]
                save_fp = f"{save_dir}/{fn}.{args.save_format}"
                latent_save_fp = f"{latent_dir}/{fn}.npy"

            # save videos
            if len(videos):
                save_videos(videos[j], save_fp, fps=args.fps / args.frame_interval)
                logger.info(f"Video saved in {save_fp}")

            # save decoded latents
            if args.save_latent:
                np.save(latent_save_fp, latents[j : j + 1])
                logger.info(f"Denoised latents saved in {latent_save_fp}")


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
        "--model_version", default="v1", type=str, choices=["v1", "v1.1", "v1.2"], help="OpenSora model version."
    )
    parser.add_argument("--image_size", type=int, nargs="+", help="image size in [256, 512]")
    parser.add_argument("--resolution", type=str, help=f"Supported video resolutions: {list(ASPECT_RATIOS.keys())}")
    parser.add_argument(
        "--aspect_ratio", type=str, help=f"Supported video aspect ratios: {list(ASPECT_RATIO_MAP.keys())}"
    )
    parser.add_argument("--num_frames", type=str, default="16", help="number of frames")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="number of videos to be generated unconditionally. If using text or class as conditions,"
        " the number of samples will be defined by the number of class labels or text captions",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="latte checkpoint path. If specified, will load from it, otherwise, will use random initialization",
    )
    parser.add_argument("--t5_model_dir", default=None, type=str, help="the T5 cache folder path")
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-ema.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )
    parser.add_argument(
        "--vae_type",
        type=str,
        choices=["OpenSora-VAE-v1.2", "VideoAutoencoderKL"],
        help="If None, use VideoAutoencoderKL, which is a spatial VAE from SD, for opensora v1.0 and v1.1. \
                If OpenSora-VAE-v1.2, will use 3D VAE (spatial + temporal), typically for opensora v1.2",
    )
    parser.add_argument(
        "--vae_micro_batch_size",
        type=int,
        default=None,
        help="If not None, split batch_size*num_frames into smaller ones for VAE encoding to reduce memory limitation. Used by spatial vae",
    )
    parser.add_argument(
        "--vae_micro_frame_size",
        type=int,
        default=17,
        help="If not None, split batch_size*num_frames into smaller ones for VAE encoding to reduce memory limitation. Used by temporal vae",
    )
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports [“O0”, “O1”, “O2”]."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument("--sampling_steps", type=int, default=50, help="Diffusion Sampling Steps")
    parser.add_argument("--guidance_scale", type=float, default=8.5, help="the scale for classifier-free guidance")
    parser.add_argument(
        "--guidance_channels",
        type=int,
        help="How many channels to use for classifier-free diffusion. If None, use half of the latent channels",
    )
    parser.add_argument(
        "--frame_interval",
        default=1,
        type=int,
        help="Frames sampling frequency. Final video FPS will be equal to FPS / frame_interval.",
    )
    parser.add_argument("--loop", type=int, default=1, help="Number of times to loop video generation task.")
    parser.add_argument("--model_max_length", type=int, default=120, help="T5's embedded sequence length.")
    parser.add_argument(
        "--condition_frame_length",
        type=int,
        help="Number of frames generated in a previous loop to use as a conditioning for the next loop.",
    )
    parser.add_argument(
        "--condition_frame_edit",
        type=float,
        help="The intensity of editing conditioning frames, where 0 means no edit and 1 means a complete edit.",
    )
    parser.add_argument(
        "--mask_strategy", type=str, nargs="*", help="Masking strategy for Image/Video-to-Video generation task."
    )
    parser.add_argument(
        "--reference_path", type=str, nargs="*", help="References for Image/Video-to-Video generation task."
    )
    # MS new args
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--debug", type=str2bool, default=False, help="Execute inference in debug mode.")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument(
        "--patchify",
        type=str,
        default="conv2d",
        choices=["conv3d", "conv2d", "linear"],
        help="patchify_conv3d_replace, conv2d - equivalent conv2d to replace conv3d patchify, linear - equivalent linear layer to replace conv3d patchify  ",
    )
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    parser.add_argument(
        "--enable_sequence_parallelism",
        default=False,
        type=str2bool,
        help="whether to enable sequence parallelism. Default is False",
    )
    parser.add_argument("--dsp", default=False, type=str2bool, help="Use DSP instead of SP in sequence parallel.")
    parser.add_argument(
        "--dtype",
        default="fp32",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp32`, which corresponds to ms.float32",
    )
    parser.add_argument(
        "--vae_dtype",
        default="fp32",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for VAE. Default is `fp32`, which corresponds to ms.float32",
    )
    parser.add_argument(
        "--t5_dtype",
        default="fp32",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for T5 model. Default is `fp32`, which corresponds to ms.float32",
    )
    parser.add_argument(
        "--amp_level",
        default="O2",
        type=str,
        help="mindspore amp level, O1: most fp32, only layers in whitelist compute in fp16 (dense, conv, etc); \
            O2: most fp16, only layers in blacklist compute in fp32 (batch norm etc)",
    )
    parser.add_argument("--space_scale", default=0.5, type=float, help="stdit model space scalec")
    parser.add_argument("--time_scale", default=1.0, type=float, help="stdit model time scalec")
    parser.add_argument(
        "--captions",
        type=str,
        nargs="+",
        help="A list of text captions to be generated with",
    )
    parser.add_argument("--prompt_path", default=None, type=str, help="path to a csv file containing captions")
    parser.add_argument(
        "--output_path",
        type=str,
        default="samples",
        help="output dir to save the generated videos",
    )
    parser.add_argument(
        "--append_timestr",
        type=str2bool,
        default=True,
        help="If true, an subfolder named with timestamp under output_path will be created to save the sampling results",
    )
    parser.add_argument(
        "--save_format",
        default="mp4",
        choices=["gif", "mp4", "png"],
        type=str,
        help="video format for saving the sampling output: gif, mp4 or png",
    )
    parser.add_argument("--fps", type=int, default=8, help="FPS in the saved video")
    parser.add_argument("--batch_size", default=4, type=int, help="infer batch size")
    parser.add_argument("--text_embed_folder", type=str, default=None, help="path to t5 embedding")
    parser.add_argument(
        "--save_latent",
        type=str2bool,
        default=False,
        help="Save denoised video latent. If True, the denoised latents will be saved in $output_path/denoised_latents",
    )
    parser.add_argument(
        "--use_vae_decode",
        type=str2bool,
        default=True,
        help="[For T2V models only] If False, skip vae decode to save memory"
        " (you can use infer_vae_decode.py to decode the saved denoised latent later.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim", "rflow"],
        help="Which sampling technique to use.",
    )
    parser.add_argument("--pre_patchify", default=False, type=str2bool, help="Patchify the latent before inference.")
    parser.add_argument("--max_image_size", default=512, type=int, help="Max image size for patchified latent.")
    parser.add_argument("--max_num_frames", default=16, type=int, help="Max number of frames for patchified latent.")
    default_args = parser.parse_args()

    __dir__ = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(__dir__, ".."))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = to_abspath(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    # convert to absolute path, necessary for modelarts
    args.ckpt_path = to_abspath(abs_path, args.ckpt_path)
    args.vae_checkpoint = to_abspath(abs_path, args.vae_checkpoint)
    args.prompt_path = to_abspath(abs_path, args.prompt_path)
    args.output_path = to_abspath(abs_path, args.output_path)
    args.text_embed_folder = to_abspath(abs_path, args.text_embed_folder)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
