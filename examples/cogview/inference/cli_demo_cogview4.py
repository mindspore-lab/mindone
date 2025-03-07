"""
Running the Script:

```bash
python generate_image.py --prompt "A serene landscape with mountains and a river" --width 1280 --height 720
```

Additional arguments include:
•	--model_path: Path to the pre-trained model (e.g., THUDM/CogView4-6B).
•	--guidance_scale: The classifier-free guidance scale for enhancing the generated image quality.
•	--num_images_per_prompt: Number of images to generate for each prompt.
•	--num_inference_steps: The number of denoising steps during inference.
•	--output_path: Path to save the generated image.
•	--dtype: Precision type for inference (either bfloat16 or float32，float16 will cause error with NaN).

This version focuses only on the relevant arguments and features of the script, without additional details about non-existent fields.

"""

import argparse

import mindspore as ms

from mindone.diffusers import CogView4Pipeline


def generate_image(
    prompt, model_path, guidance_scale, num_images_per_prompt, num_inference_steps, width, height, output_path, dtype
):
    # Load the pre-trained model with the specified precision
    pipe = CogView4Pipeline.from_pretrained(model_path, mindspore_dtype=dtype)

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    # Generate the image based on the prompt
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
    )[0][0]

    # Save the generated image to the local file system
    image.save(output_path)

    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image using the CogView4 model.")

    # Define arguments for prompt, model path, etc.
    parser.add_argument("--prompt", type=str, required=True, help="The text description for generating the image.")
    parser.add_argument("--model_path", type=str, default="THUDM/CogView4-6B", help="Path to the pre-trained model.")
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5, help="The guidance scale for classifier-free guidance."
    )
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images to generate per prompt.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps for inference.")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=720, help="Height of the generated image.")
    parser.add_argument("--output_path", type=str, default="cogview4.png", help="Path to save the generated image.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Precision type (float16 or float32).")

    # Parse the arguments
    args = parser.parse_args()

    # Convert dtype argument to mindspore dtype
    dtype = ms.bfloat16 if args.dtype == "bfloat16" else ms.float32

    # Call the function to generate the image
    generate_image(
        prompt=args.prompt,
        model_path=args.model_path,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        num_inference_steps=args.num_inference_steps,
        width=args.width,
        height=args.height,
        output_path=args.output_path,
        dtype=dtype,
    )
