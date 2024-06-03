import datetime
import logging
import os
import sys

import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from inference import init_env, data_parallel_split
from opensora.models.stdit import STDiT2_XL_2
from opensora.models.text_encoder.t5 import get_text_encoder_and_tokenizer
from opensora.models.vae.vae import SD_CONFIG, AutoencoderKL
from opensora.pipelines import InferPipeline
from opensora.utils.amp import auto_mixed_precision
from opensora.utils.cond_data import get_references, read_captions_from_csv, read_captions_from_txt
from opensora.utils.model_utils import WHITELIST_OPS
from opensora.utils.util import apply_mask_strategy, process_mask_strategies, process_prompts

from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)


def main(args):
    if args.append_timestr:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir = f"{args.output_path}/{time_str}"
    else:
        save_dir = f"{args.output_path}"

    os.makedirs(save_dir, exist_ok=True)
    if args.save_latent:
        latent_dir = os.path.join(args.output_path, "denoised_latents")
        os.makedirs(latent_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    # TODO: add distributed support
    rank_id, device_num = init_env(
        args.mode,
        args.seed,
        args.use_parallel,
        device_target=args.device_target,
        enable_dvm=args.enable_dvm,
        debug=args.debug,
    )
    set_random_seed(args.seed)

    # get captions from cfg or prompt_path
    if args.prompt_path is not None:
        if args.prompt_path.endswith(".csv"):
            captions = read_captions_from_csv(args.prompt_path)
        elif args.prompt_path.endswith(".txt"):
            captions = read_captions_from_txt(args.prompt_path)
    else:
        captions = args.captions
    captions = process_prompts(captions, args.loop)

    # split for data parallel
    captions, base_data_idx = data_parallel_split(captions, rank_id, device_num)
    print(f"Num captions for rank {rank_id}: {len(captions)}")

    # 2. model initiate and weight loading
    # 2.1 latte
    logger.info("STDiT2 init")

    VAE_T_COMPRESS = 1
    VAE_S_COMPRESS = 8
    VAE_Z_CH = SD_CONFIG["z_channels"]
    img_h, img_w = args.image_size if isinstance(args.image_size, list) else (args.image_size, args.image_size)
    input_size = (
        args.num_frames // VAE_T_COMPRESS,
        img_h // VAE_S_COMPRESS,
        img_w // VAE_S_COMPRESS,
    )
    if args.image_size == 512 and args.space_scale == 0.5:
        logger.warning("space_ratio should be 1 for 512x512 resolution")
    model_extra_args = dict(
        input_size=input_size,
        in_channels=VAE_Z_CH,
        model_max_length=args.model_max_length,
        patchify_conv3d_replace=args.patchify,  # for Ascend
        enable_flashattn=args.enable_flash_attention,
        input_sq_size=512,
        qk_norm=True,
    )
    latte_model = STDiT2_XL_2(**model_extra_args)
    latte_model = latte_model.set_train(False)

    dtype_map = {"fp16": ms.float16, "bf16": ms.bfloat16}
    if args.dtype in ["fp16", "bf16"]:
        latte_model = auto_mixed_precision(
            latte_model, amp_level=args.amp_level, dtype=dtype_map[args.dtype], custom_fp32_cells=WHITELIST_OPS
        )

    if len(args.ckpt_path) > 0:
        logger.info(f"Loading ckpt {args.ckpt_path} into STDiT")
        assert os.path.exists(args.ckpt_path), f"{args.ckpt_path} not found."
        latte_model.load_from_checkpoint(args.ckpt_path)
    else:
        logger.warning("STDiT uses random initialization!")

    # 2.2 vae
    if args.use_vae_decode:
        logger.info("vae init")
        vae = AutoencoderKL(
            SD_CONFIG,
            VAE_Z_CH,
            ckpt_path=args.vae_checkpoint,
        )
        vae = vae.set_train(False)
        if args.vae_dtype in ["fp16", "bf16"]:
            vae = auto_mixed_precision(
                vae, amp_level=args.amp_level, dtype=dtype_map[args.vae_dtype], custom_fp32_cells=[nn.GroupNorm]
            )
    else:
        vae = None

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
        if args.dtype in dtype_map.keys():
            text_encoder = auto_mixed_precision(text_encoder, amp_level="O2", dtype=dtype_map[args.dtype])
    else:
        raise NotImplementedError("Embedded tokens are not supported yet")
    assert num_prompts > 0, "No captions provided"
    logger.info(f"Num tokens: {mask.asnumpy().sum(2)}")

    # 3. build inference pipeline
    pipeline = InferPipeline(
        latte_model,
        vae,
        text_encoder=text_encoder,
        scale_factor=args.sd_scale_factor,
        num_inference_steps=args.sampling_steps,
        guidance_rescale=args.guidance_scale,
        guidance_channels=args.guidance_channels,
        ddim_sampling=False,  # TODO: add ddim support
        condition="text",
        micro_batch_size=args.vae_micro_batch_size,
    )

    # 3.1. support for multi-resolution
    model_args = dict()
    model_args["height"] = Tensor([img_h] * args.batch_size, dtype=ms.float32)
    model_args["width"] = Tensor([img_w] * args.batch_size, dtype=ms.float32)
    model_args["num_frames"] = Tensor([args.num_frames] * args.batch_size, dtype=ms.float32)
    model_args["ar"] = Tensor([img_h / img_w] * args.batch_size, dtype=ms.float32)
    model_args["fps"] = Tensor([args.fps] * args.batch_size, dtype=ms.float32)

    # 3.2 reference
    print('D--: ', args.reference_path)
    if args.reference_path is not None:
        assert not args.use_parallel, "parallel inference is not supported for I2V"
        assert len(args.reference_path) == len(
            captions
        ), f"Reference path mismatch: {len(args.reference_path)} != {len(captions)}"
        assert len(args.reference_path) == len(
            args.mask_strategy
        ), f"Mask strategy mismatch: {len(args.mask_strategy)} != {len(captions)}"
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
            f"Sampling steps {args.sampling_steps}",
            f"DDIM sampling: {args.ddim_sampling}",
            f"CFG guidance scale: {args.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    for i in range(0, num_prompts, args.batch_size):
        batch_prompts = captions[i : i + args.batch_size]
        ns = len(batch_prompts)
        frames_mask_strategy = frames_mask_strategies[i : i + args.batch_size]

        references = get_references(args.reference_path[i : i + args.batch_size], (img_h, img_w))
        # embed references into latent space
        for ref in references:
            if ref is not None:
                for k in range(len(ref)):
                    ref[k] = pipeline.vae_encode(Tensor(ref[k])).asnumpy().swapaxes(0, 1)

        latents, videos = None, []
        for loop_i in range(args.loop):
            if loop_i > 0:
                for j in range(len(references)):  # iterate over batch of references
                    if references[j] is None:
                        references[j] = [latents[j]]
                    else:
                        references[j].append(latents[j])
                    new_strategy = [
                        loop_i,
                        len(references[j]) - 1,
                        -args.condition_frame_length,
                        0,
                        args.condition_frame_length,
                        0.0,
                    ]
                    if frames_mask_strategy[j] is None:
                        frames_mask_strategy[j] = [new_strategy]
                    else:
                        frames_mask_strategy[j].append(new_strategy)

            # prepare inputs
            inputs = {}
            # b c t h w
            z = np.random.randn(*([ns, VAE_Z_CH] + list(input_size))).astype(np.float32)

            z, frames_mask = apply_mask_strategy(z, references, frames_mask_strategy, loop_i)
            frames_mask = Tensor(frames_mask, dtype=ms.float32)

            z = ms.Tensor(z, dtype=ms.float32)

            inputs["noise"] = z
            inputs["scale"] = args.guidance_scale
            inputs["text_tokens"] = text_tokens[i : i + ns, loop_i]
            inputs["text_emb"] = None
            inputs["mask"] = mask[i : i + ns, loop_i]

            logger.info("Sampling for captions: ")
            for j in range(ns):
                logger.info(captions[i + j][loop_i])

            # infer
            samples, latents = pipeline(inputs, frames_mask=frames_mask, additional_kwargs=model_args)
            samples, latents = samples.asnumpy(), latents.asnumpy()
            videos.append(samples[:, args.condition_frame_length if loop_i > 0 else 0 :])

        videos = np.concatenate(videos, axis=1)

        # save result
        for j in range(ns):
            global_idx = base_data_idx + i + j
            if args.text_embed_folder is None:
                prompt = "-".join((batch_prompts[j][0].replace("/", "").split(" ")[:10]))
                save_fp = f"{save_dir}/{global_idx:03d}-{prompt}.{args.save_format}"
            else:
                fn = prompt_prefix[global_idx]
                save_fp = f"{save_dir}/{fn}.{args.save_format}"
            # save videos
            if videos is not None:
                save_videos(videos[j : j + 1], save_fp, fps=args.fps / args.frame_interval)
                logger.info(f"Video saved in {save_fp}")


if __name__ == "__main__":
    from inference import parse_args

    args = parse_args()
    main(args)
