# Adapted from https://github.com/VectorSpaceLab/OmniGen2/blob/main/inference.py
import argparse
import os
from typing import Union

import numpy as np
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2 import OmniGen2ChatPipeline, OmniGen2Pipeline
from omnigen2.schedulers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
from omnigen2.utils.img_util import create_collage
from PIL import Image, ImageOps
from transformers import Qwen2_5_VLProcessor

from mindspore import dtype
from mindspore.nn import no_init_parameters

from mindone.diffusers import AutoencoderKL
from mindone.transformers import Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OmniGen2 image generation script.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--chat_mode", action="store_true", help="Enable chat mode.")
    parser.add_argument("--transformer_path", type=str, default=None, help="Path to transformer checkpoint.")
    parser.add_argument("--transformer_lora_path", type=str, default=None, help="Path to transformer LoRA checkpoint.")
    parser.add_argument(
        "--scheduler", type=str, default="euler", choices=["euler", "dpmsolver++"], help="Scheduler to use."
    )
    parser.add_argument("--num_inference_step", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument(
        "--max_input_image_pixels", type=int, default=1048576, help="Maximum number of pixels for each input image."
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Data type for model weights."
    )
    parser.add_argument("--text_guidance_scale", type=float, default=5.0, help="Text guidance scale.")
    parser.add_argument("--image_guidance_scale", type=float, default=2.0, help="Image guidance scale.")
    parser.add_argument("--cfg_range_start", type=float, default=0.0, help="Start of the CFG range.")
    parser.add_argument("--cfg_range_end", type=float, default=1.0, help="End of the CFG range.")
    parser.add_argument(
        "--instruction", type=str, default="A dog running in the park", help="Text prompt for generation."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated,"
        " (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Negative prompt for generation.",
    )
    parser.add_argument("--input_image_path", type=str, nargs="+", default=None, help="Path(s) to input image(s).")
    parser.add_argument("--output_image_path", type=str, default="output.png", help="Path to save output image.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images to generate per prompt.")
    parser.add_argument("--enable_model_cpu_offload", action="store_true", help="Enable model CPU offload.")
    parser.add_argument("--enable_sequential_cpu_offload", action="store_true", help="Enable sequential CPU offload.")
    parser.add_argument("--enable_group_offload", action="store_true", help="Enable group offload.")
    parser.add_argument("--enable_teacache", action="store_true", help="Enable teacache to speed up inference.")
    parser.add_argument(
        "--teacache_rel_l1_thresh", type=float, default=0.05, help="Relative L1 threshold for teacache."
    )
    parser.add_argument("--enable_taylorseer", action="store_true", help="Enable TaylorSeer Caching.")
    return parser.parse_args()


@no_init_parameters()
def load_pipeline(args: argparse.Namespace, weight_dtype: dtype.Type) -> Union[OmniGen2Pipeline, OmniGen2ChatPipeline]:
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, subfolder="mllm", mindspore_dtype=weight_dtype
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path, subfolder="mllm_processor")
    if args.scheduler == "dpmsolver++":
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++", solver_type="midpoint", solver_order=2, prediction_type="flow_prediction"
        )
    else:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.transformer_path or args.model_path,
        subfolder=None if args.transformer_path else "transformer",
        mindspore_dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae", mindspore_dtype=weight_dtype)

    if args.chat_mode:
        pipeline = OmniGen2ChatPipeline(transformer, vae, scheduler, mllm, processor)

        if args.transformer_lora_path:
            print("WARNING: LoRA weights are not supported for chat mode.")
        if args.enable_teacache or args.enable_taylorseer:
            print("WARNING: inference caching is not supported for chat mode.")
    else:
        pipeline = OmniGen2Pipeline(transformer, vae, scheduler, mllm, processor)

        if args.transformer_lora_path:
            print(f"LoRA weights loaded from {args.transformer_lora_path}")
            pipeline.load_lora_weights(args.transformer_lora_path)

        if args.enable_teacache and args.enable_taylorseer:
            print(
                "WARNING: enable_teacache and enable_taylorseer are mutually exclusive. enable_teacache will be ignored."
            )

        if args.enable_taylorseer:
            pipeline.enable_taylorseer = True
        elif args.enable_teacache:
            pipeline.transformer.enable_teacache = True
            pipeline.transformer.teacache_rel_l1_thresh = args.teacache_rel_l1_thresh

    return pipeline


def preprocess(input_image_path: list[str]) -> tuple[str, str, list[Image.Image]]:
    """Preprocess the input images."""
    # Process input images
    input_images = None

    if input_image_path:
        if isinstance(input_image_path, str):
            input_image_path = [input_image_path]

        if len(input_image_path) == 1 and os.path.isdir(input_image_path[0]):
            input_images = [
                Image.open(os.path.join(input_image_path[0], f)).convert("RGB") for f in os.listdir(input_image_path[0])
            ]
        else:
            input_images = [Image.open(path).convert("RGB") for path in input_image_path]

        input_images = [ImageOps.exif_transpose(img) for img in input_images]

    return input_images


def run(
    args: argparse.Namespace,
    pipeline: Union[OmniGen2Pipeline, OmniGen2ChatPipeline],
    instruction: str,
    negative_prompt: str,
    input_images: list[Image.Image],
) -> Image.Image:
    """Run the image generation pipeline with the given parameters."""
    generator = np.random.default_rng(args.seed)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_step,
        max_sequence_length=1024,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        cfg_range=(args.cfg_range_start, args.cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )
    return results


def main(args: argparse.Namespace) -> None:
    """Main function to run the image generation process."""
    # Set weight dtype
    weight_dtype = dtype.float32
    if args.dtype == "fp16":
        weight_dtype = dtype.float16
    elif args.dtype == "bf16":
        weight_dtype = dtype.bfloat16

    # Load pipeline and process inputs
    pipeline = load_pipeline(args, weight_dtype)
    input_images = preprocess(args.input_image_path)

    # Generate and save image
    results = run(args, pipeline, args.instruction, args.negative_prompt, input_images)

    if results.images is not None:
        os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)
        if len(results.images) > 1:
            for i, image in enumerate(results.images):
                image_name, ext = os.path.splitext(args.output_image_path)
                image.save(f"{image_name}_{i}{ext}")

        vis_images = [np.array(image, dtype=np.float32) / 127.5 - 1 for image in results.images]
        output_image = create_collage(vis_images)

        output_image.save(args.output_image_path)
        print(f"Image saved to {args.output_image_path}")

    if results.text is not None:
        print(f"Text: {results.text}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
