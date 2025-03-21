from argparse import ArgumentParser

from omnigen import OmniGenPipeline

import mindspore as ms


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Shitao/OmniGen-v1",
        help="Model ID from huggingface hub or local path",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A grass-type Pokemon in a forest",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./pokemon.png",
        help="Path to save generated image",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of generated image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of generated image",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Guidance scale for classifier-free guidance",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipe = OmniGenPipeline.from_pretrained(args.model_id)
    pipe.merge_lora(args.lora_path)
    images = pipe(
        prompt=args.prompt, height=args.height, width=args.width, guidance_scale=args.guidance_scale, dtype=ms.bfloat16
    )
    images[0].save(args.output_path)


if __name__ == "__main__":
    main()
