import argparse
import os
import time

import numpy as np

import mindspore as ms

from examples.accelerated_dit_pipelines.FBCache_ms.pipeline_flux import FluxPipeline

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt/FLUX.1-dev",
        help="Flux model checkpoint",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        nargs="+",
        type=str,
        default="A cat holding a sign that says hello world",
        help="Prompt, do not support prompt list",
    )
    parser.add_argument("--negative_prompt", type=str, nargs="+", default="", help="Negative Prompt")
    parser.add_argument(
        "--image_size", type=int, nargs="+", default=[1024, 1024], help="Output image size (height, width)"
    )
    parser.add_argument("--save_path", type=str, default="./", help="Output save path")
    parser.add_argument("--use_graph_mode", action="store_true", help="Use graph mode to accelerate.")
    parser.add_argument("--load_ckpt", action="store_true", help="Load checkpoint")
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="Guidance scale")
    parser.add_argument("--residual_diff_threshold", type=float, default=0.12, help="FBCache diff threshold")
    parser.add_argument("--save_params", action="store_true", help="Save cache usage history")
    parser.add_argument(
        "--save_params_path",
        type=str,
        default=os.path.join(parent_path, "config/cache_usage_history.json"),
        help="Save cache usage history path",
    )
    parser.add_argument("--taylorseer_derivative", type=int, default=0, help="Taylorseer derivative order")

    args = parser.parse_args()
    return args


def warmup(pipe, args):
    height, width = args.image_size
    pipe(
        prompt="0",
        negative_prompt=args.negative_prompt,
        width=width,
        height=height,
        guidance_scale=args.guidance_scale,
        taylorseer_derivative=args.taylorseer_derivative,
    )


def main():
    args = arg_parse()
    if args.use_graph_mode:
        ms.set_context(mode=0, jit_config={"jit_level": "O1"})

    if args.load_ckpt:
        pipe = FluxPipeline.from_pretrained(
            args.ckpt,
            mindspore_dtype=ms.bfloat16,
            custom_pipeline=os.path.join(parent_path, "pipeline_flux.py"),
        )
    else:
        from transformers import CLIPTokenizer, T5TokenizerFast
        from transformers.models.clip.configuration_clip import CLIPTextConfig
        from transformers.models.t5.configuration_t5 import T5Config

        from mindone.diffusers.models.autoencoders import AutoencoderKL
        from mindone.diffusers.models.transformers import FluxTransformer2DModel
        from mindone.diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from mindone.transformers import CLIPTextModel, T5EncoderModel

        scheduler_config = FlowMatchEulerDiscreteScheduler.load_config(args.ckpt, subfolder="scheduler")
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        vae_config = AutoencoderKL.load_config(args.ckpt, subfolder="vae")
        vae = AutoencoderKL.from_config(vae_config)
        text_encoder_config = CLIPTextConfig.from_json_file(f"{args.ckpt}/text_encoder/config.json")
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained(args.ckpt, subfolder="tokenizer")
        text_encoder2 = T5EncoderModel(T5Config.from_json_file(f"{args.ckpt}/text_encoder_2/config.json"))
        tokenizer2 = T5TokenizerFast.from_pretrained(args.ckpt, subfolder="tokenizer_2")
        transformer_config = FluxTransformer2DModel.load_config(args.ckpt, subfolder="transformer")
        transformer = FluxTransformer2DModel.from_config(transformer_config)

        pipe = FluxPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder2,
            tokenizer_2=tokenizer2,
            transformer=transformer,
        ).to(ms.bfloat16)
        pipe.register_to_config(_name_or_path=args.ckpt)

    print("Start warmup...")
    warmup(pipe, args)

    print("Start generate >>>", flush=True)
    start = time.time()
    height, width = args.image_size
    generator = np.random.Generator(np.random.PCG64(0))
    imgs = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=width,
        height=height,
        guidance_scale=args.guidance_scale,
        generator=generator,
        taylorseer_derivative=args.taylorseer_derivative,
    )[0]
    print(f"generate img spend time: {(time.time()-start):.2f}s")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for i, img in enumerate(imgs):
        img.save(os.path.join(args.save_path, f"output_{i}.png"))


if __name__ == "__main__":
    main()
