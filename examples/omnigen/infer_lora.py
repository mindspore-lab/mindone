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
    return parser.parse_args()


def main():
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
    # pipe.merge_lora(args.lora_path)
    images = pipe(
        prompt="a photo of dragon flying with sks dog face", height=512, width=512, guidance_scale=3, dtype=ms.bfloat16
    )
    images[0].save("example_sks_dog_snow.png")


if __name__ == "__main__":
    main()
