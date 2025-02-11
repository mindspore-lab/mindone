"""
PixArt Model Inference Script with LoRA Weights

This script demonstrates how to:
1. Load a pretrained PixArt model
2. Load and apply LoRA weights
3. Generate images using the fine-tuned model
"""

import argparse

import mindspore as ms

from mindone.diffusers import PixArtAlphaPipeline, Transformer2DModel
from mindone.diffusers._peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="PixArt-alpha/PixArt-XL-2-512x512",
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the base transformer model in FP16 for memory efficiency
    transformer = Transformer2DModel.from_pretrained(args.model_id, subfolder="transformer", mindspore_dtype=ms.float16)

    # Load LoRA weights and apply them to the transformer
    transformer = PeftModel.from_pretrained(transformer, args.lora_path)

    # Initialize the PixArt pipeline with our LoRA-enhanced transformer
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_id,
        transformer=transformer,  # Use our modified transformer
        mindspore_dtype=ms.float16,  # Keep everything in FP16
    )

    # Generate an image using the text prompt
    image = pipe(args.prompt)[0][0]

    # Save the generated image
    image.save(args.output_path)


if __name__ == "__main__":
    main()
