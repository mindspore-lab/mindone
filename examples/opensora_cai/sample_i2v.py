import datetime
import logging
import os

import numpy as np
from opensora.models import STDiT2_XL_2
from opensora.models.autoencoder import SD_CONFIG, AutoencoderKL
from opensora.models.layers.blocks import Attention, LayerNorm
from opensora.models.text_encoders import get_text_encoder_and_tokenizer
from opensora.pipelines import InferPipeline
from sample_t2v import init_env, read_captions_from_csv
from utils import apply_mask_strategy, get_references, process_mask_strategies, process_prompts

import mindspore as ms
from mindspore import Tensor, nn

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    init_env(args.mode, args.device_target, args.enable_dvm)
    set_random_seed(args.seed)

    # get captions from cfg or prompt_file
    if args.prompt_file is not None:
        if args.prompt_file.endswith(".csv"):
            captions = read_captions_from_csv(args.prompt_file)
        else:  # any other text format file
            captions = []
            with open(args.caption_file, "r") as fp:
                for line in fp:
                    captions.append(line.strip())
    else:
        captions = args.captions

    captions = process_prompts(captions, args.loop)

    dtype_map = {"fp16": ms.float16, "bf16": ms.bfloat16}

    # 2.3 text encoder
    if args.embed_path is None:
        text_encoder, tokenizer = get_text_encoder_and_tokenizer(
            "t5", args.t5_model_dir, model_max_length=args.model_max_length
        )
        n = len(captions)
        text_tokens, mask = zip(
            *[text_encoder.get_text_tokens_and_mask(caption, return_tensor=False) for caption in captions]
        )
        text_tokens, mask = Tensor(text_tokens, dtype=ms.int32), Tensor(mask, dtype=ms.uint8)
        if args.dtype in dtype_map.keys():
            text_encoder = auto_mixed_precision(text_encoder, amp_level="O2", dtype=dtype_map[args.dtype])
    else:
        raise NotImplementedError("Embedded tokens are not supported yet")
    assert n > 0, "No captions provided"
    logger.info(f"Num tokens: {mask.asnumpy().sum(2)}")

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
        model_max_length=text_encoder.model_max_length,
        patchify_conv3d_replace="conv2d",  # for Ascend
        enable_flashattn=args.enable_flash_attention,
        input_sq_size=512,
        qk_norm=True,
    )
    latte_model = STDiT2_XL_2(**model_extra_args)
    latte_model = latte_model.set_train(False)

    if args.dtype in ["fp16", "bf16"]:
        latte_model = auto_mixed_precision(
            latte_model,
            amp_level=args.amp_level,
            dtype=dtype_map[args.dtype],
            custom_fp32_cells=[LayerNorm, Attention, nn.SiLU, nn.GELU],
            # NOTE: keep it the same as training setting
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
    if args.vae_dtype in ["fp16", "bf16"]:
        vae = auto_mixed_precision(vae, amp_level=args.amp_level, dtype=dtype_map[args.vae_dtype])

    # 3.4. support for multi-resolution
    model_args = dict()
    model_args["height"] = Tensor([img_h] * args.batch_size, dtype=ms.float32)
    model_args["width"] = Tensor([img_w] * args.batch_size, dtype=ms.float32)
    model_args["num_frames"] = Tensor([args.num_frames] * args.batch_size, dtype=ms.float32)
    model_args["ar"] = Tensor([img_h / img_w] * args.batch_size, dtype=ms.float32)
    model_args["fps"] = Tensor([args.fps] * args.batch_size, dtype=ms.float32)

    # 3.5 reference
    if args.reference_path is not None:
        assert len(args.reference_path) == len(
            captions
        ), f"Reference path mismatch: {len(args.reference_path)} != {len(captions)}"
        assert len(args.reference_path) == len(
            args.mask_strategy
        ), f"Mask strategy mismatch: {len(args.mask_strategy)} != {len(captions)}"
    else:
        args.reference_path = [None] * len(captions)
        args.mask_strategy = [None] * len(captions)

    # 3. build inference pipeline
    pipeline = InferPipeline(
        latte_model,
        vae,
        text_encoder=text_encoder,
        scale_factor=args.sd_scale_factor,
        num_inference_steps=args.sampling_steps,
        guidance_rescale=args.guidance_scale,
        ddim_sampling=False,  # FIXME
        condition="text",
    )

    frames_mask_strategies = process_mask_strategies(args.mask_strategy)

    # 4.1. batch generation
    for i in range(0, len(captions), args.batch_size):
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
            samples, latents = pipeline(
                inputs, frames_mask=frames_mask, additional_kwargs=model_args, return_latents=True
            )
            samples, latents = samples.asnumpy(), latents.asnumpy()
            videos.append(samples[:, args.condition_frame_length if loop_i > 0 else 0 :])

        videos = np.concatenate(videos, axis=1)

        # save result
        for j in range(ns):
            global_idx = i * args.batch_size + j
            prompt = "-".join((batch_prompts[j][0].replace("/", "").split(" ")[:10]))
            save_fp = f"{save_dir}/{global_idx:03d}-{prompt}.{args.save_format}"
            save_videos(videos[j], save_fp, fps=args.fps)
            logger.info(f"save to {save_fp}")


if __name__ == "__main__":
    from sample_t2v import parse_args

    args = parse_args()
    main(args)
