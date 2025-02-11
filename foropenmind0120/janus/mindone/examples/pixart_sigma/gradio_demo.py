"""
A text-to-image generation system using PixArt-Sigma model and MindSpore framework.
This script provides both CLI and web interface for generating images from text prompts.
The system uses transformer-based architecture with features like resolution binning,
mixed-precision training, and KV compression for optimization.
"""

# Core system imports
import argparse
import logging
import os
import random

import gradio as gr
import numpy as np

# Resolution binning constants and utilities
from pixart.dataset import (
    ASPECT_RATIO_256_BIN,
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
    ASPECT_RATIO_2048_BIN,
    classify_height_width_bin,
)
from pixart.diffusers import AutoencoderKL
from pixart.modules.pixart import PixArt_XL_2, PixArtMS_XL_2

# PixArt model and pipeline imports
from pixart.pipelines.infer_pipeline import PixArtInferPipeline
from pixart.utils import (
    create_save_func,
    init_env,
    load_ckpt_params,
    organize_prompts,
    resize_and_crop_tensor,
    str2bool,
)
from tqdm import trange
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import ops

from mindone.transformers import T5EncoderModel
from mindone.utils.amp import auto_mixed_precision

# Setup logging and define maximum seed value for reproducibility
logger = logging.getLogger(__name__)
MAX_SEED = np.iinfo(np.int32).max


def parse_args():
    """
    Command line argument parser for configuring the PixArt pipeline.
    Includes model settings, optimization parameters, and output configurations.
    """
    parser = argparse.ArgumentParser()
    # Model Configuration:
    # --sample_size: Controls the size of latent representations (32, 64, or 128)
    # --image_height/width: Output image dimensions
    # --use_resolution_binning: Enables smart resolution selection for better quality
    parser.add_argument("--sample_size", default=128, type=int, choices=[128, 64, 32])
    parser.add_argument("--image_height", default=1024, type=int)
    parser.add_argument("--image_width", default=1024, type=int)
    parser.add_argument("--use_resolution_binning", default=True, type=str2bool)

    # Model paths and weights
    parser.add_argument("--checkpoint", default="models/PixArt-Sigma-XL-2-1024-MS.ckpt")
    parser.add_argument("--vae_root", default="models/vae")
    parser.add_argument("--tokenizer_root", default="models/tokenizer")
    parser.add_argument("--text_encoder_root", default="models/text_encoder")

    # Model scaling and optimization parameters
    parser.add_argument("--sd_scale_factor", default=0.13025, type=float)
    parser.add_argument("--enable_flash_attention", default=True, type=str2bool)
    parser.add_argument("--mode", default=0, choices=[0, 1], type=int)
    parser.add_argument("--jit_level", default="O1", choices=["O0", "O1"])

    # Generation control parameters
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use_parallel", default=False, type=str2bool)

    # Hardware and precision settings
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--device_target", default="Ascend", choices=["Ascend"])

    # Output configuration
    parser.add_argument("--output_path", default="./samples")

    # KV compression settings for memory optimization
    parser.add_argument("--kv_compress", default=False, type=str2bool)
    parser.add_argument(
        "--kv_compress_sampling",
        default="conv",
        choices=["conv", "ave", "uniform"],
    )
    parser.add_argument("--kv_compress_scale_factor", default=1, type=int)
    parser.add_argument("--kv_compress_layer", nargs="*", type=int)

    # Web interface and batch processing settings
    parser.add_argument("--port", default=7788, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_trials", default=1, type=int)
    parser.add_argument("--imagegrid", default=False, type=str2bool)
    return parser.parse_args()


def get_seed(seed: int, randomize_seed: bool) -> int:
    """
    Returns either a random seed or the input seed based on randomize_seed flag.
    Used for controlling reproducibility of image generation.
    """
    if randomize_seed:
        return random.randint(0, MAX_SEED)
    return seed


def get_binned_dimensions(args):
    """
    Determines optimal image dimensions based on the sample size.
    Uses predefined aspect ratio bins to maintain quality across different resolutions.
    """
    if args.use_resolution_binning:
        # Select appropriate aspect ratio bin based on sample size
        if args.sample_size == 256:
            aspect_ratio_bin = ASPECT_RATIO_2048_BIN
        elif args.sample_size == 128:
            aspect_ratio_bin = ASPECT_RATIO_1024_BIN
        elif args.sample_size == 64:
            aspect_ratio_bin = ASPECT_RATIO_512_BIN
        elif args.sample_size == 32:
            aspect_ratio_bin = ASPECT_RATIO_256_BIN
        else:
            raise ValueError(f"Invalid sample size: '{args.sample_size}'.")

        # Get optimal dimensions for the given image size
        height, width = classify_height_width_bin(args.image_height, args.image_width, ratios=aspect_ratio_bin)
        logger.info(f"{width}x{height} init")
        return height, width
    return args.image_height, args.image_width


class PixArtPipeline:
    """
    Main pipeline for text-to-image generation using PixArt model.
    Handles model initialization, generation process, and output management.
    """

    def __init__(self, args):
        """
        Initializes pipeline components including the main model, VAE,
        text encoder, and tokenizer.
        """
        self.args = args
        self.network = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.pipeline = None
        self.init_models()

    def init_models(self):
        """
        Initializes all required models with specified configuration.
        Sets up mixed precision, flash attention, and KV compression if enabled.
        """
        # Set model precision
        model_dtype = ms.float16 if self.args.dtype == "fp16" else ms.float32

        # Calculate positional encoding interpolation
        pe_interpolation = self.args.sample_size / 64
        # Choose network based on sample size
        network_fn = PixArt_XL_2 if self.args.sample_size == 32 else PixArtMS_XL_2
        # Configure KV compression if enabled
        sampling = self.args.kv_compress_sampling if self.args.kv_compress else None

        # Initialize main network with all configurations
        self.network = network_fn(
            input_size=self.args.sample_size,
            pe_interpolation=pe_interpolation,
            model_max_length=300,
            sampling=sampling,
            scale_factor=self.args.kv_compress_scale_factor,
            kv_compress_layer=self.args.kv_compress_layer,
            block_kwargs={"enable_flash_attention": self.args.enable_flash_attention},
        )

        # Apply mixed precision if using fp16
        if self.args.dtype == "fp16":
            self.network = auto_mixed_precision(self.network, amp_level="O2", dtype=model_dtype)

        # Load model weights
        self.network = load_ckpt_params(self.network, self.args.checkpoint)

        # Initialize supporting models (VAE, tokenizer, text encoder)
        self.vae = AutoencoderKL.from_pretrained(self.args.vae_root, mindspore_dtype=model_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_root, model_max_length=300)
        self.text_encoder = T5EncoderModel.from_pretrained(self.args.text_encoder_root, mindspore_dtype=model_dtype)

        # Create initial pipeline with default settings
        self.create_pipeline()

    def create_pipeline(self, sampling_method="dpm", num_inference_steps=20, guidance_scale=4.5):
        """
        Creates or updates the generation pipeline with specified parameters.
        Controls the sampling process and guidance for generation.
        """
        self.pipeline = PixArtInferPipeline(
            self.network,
            self.vae,
            self.text_encoder,
            self.tokenizer,
            scale_factor=self.args.sd_scale_factor,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            sampling_method=sampling_method,
            force_freeze=True,
        )

    def generate(self, prompt, sampling_method, sampling_steps, guidance_scale, seed=0, randomize_seed=False):
        """
        Generates images from text prompts using the configured pipeline.
        Handles the complete generation process including:
        - Pipeline parameter updates
        - Seed management
        - Dimension calculation
        - Prompt processing
        - Image generation and saving
        """
        # Update pipeline parameters if sampling method changes
        if self.pipeline.sampling_method != sampling_method:
            self.create_pipeline(sampling_method, sampling_steps, guidance_scale)
        else:
            self.pipeline.set_sampling_params(
                sampling_method=sampling_method, num_inference_steps=sampling_steps, guidance_scale=guidance_scale
            )
            self.pipeline.num_inference_steps = sampling_steps
            self.pipeline.guidance_scale = guidance_scale

        # Set seed for reproducibility
        seed = int(get_seed(seed, randomize_seed))
        ms.set_seed(seed)

        # Calculate image dimensions
        height, width = get_binned_dimensions(self.args)
        latent_height, latent_width = height // 8, width // 8

        # Process and organize prompts
        prompts = organize_prompts(
            prompts=[prompt],
            negative_prompts=[""],
            prompt_path=None,
            save_json=True,
            output_dir=self.args.output_path,
            batch_size=self.args.batch_size,
        )

        # Setup save function for generated images
        save = create_save_func(
            output_dir=self.args.output_path, imagegrid=self.args.imagegrid, grid_cols=self.args.batch_size
        )

        # Generate images
        final_output = None
        for prompt_data in prompts:
            x_samples = []
            for _ in trange(self.args.num_trials, desc="trials", disable=self.args.num_trials == 1):
                logger.info(f"Prompt(s): {prompt_data['prompt']}")
                num = len(prompt_data["prompt"])

                # Generate random latent vectors
                z = ops.randn((num, 4, latent_height, latent_width), dtype=ms.float32)
                # Generate images from prompts
                output = self.pipeline(z, prompt_data["prompt"], prompt_data["negative_prompt"]).asnumpy()
                x_samples.append(output)

            # Process and save generated images
            x_samples = np.concatenate(x_samples, axis=0)
            if self.args.use_resolution_binning:
                x_samples = resize_and_crop_tensor(x_samples, self.args.image_width, self.args.image_height)
            save(x_samples)
            final_output = x_samples[0]

        # Prepare display information
        display_info = (
            f"Model: {self.args.checkpoint}\n"
            f"Image size: {self.args.image_width}x{self.args.image_height}\n"
            f"Binned size: {width}x{height}\n"
            f"Sampling method: {sampling_method}\n"
            f"Images saved to: {self.args.output_path}"
        )

        return final_output, display_info, seed


def main():
    """
    Main function that sets up and launches the Gradio interface.
    Provides a web UI for interacting with the PixArt pipeline.
    """
    # Initialize environment and create output directory
    args = parse_args()
    _, _ = init_env(args)
    os.makedirs(args.output_path, exist_ok=True)

    # Create pipeline instance
    pixart = PixArtPipeline(args)

    # Setup Gradio interface
    title = f"PixArt-Sigma {args.image_height}px Demo"
    description = """# PixArt-Sigma MindSpore Demo
    A transformer-based text-to-image diffusion system implemented in MindSpore. English prompts only.
    """

    # Configure Gradio interface components
    demo = gr.Interface(
        fn=pixart.generate,
        inputs=[
            # Text input for prompts
            gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
            # Sampling method selection
            gr.Radio(
                choices=["iddpm", "ddim", "dpm"],
                label="Sampling Method",
                value="dpm",
            ),
            # Generation parameters
            gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=20, step=1),
            gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=4.5, step=0.1),
            # Seed control
            gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            ),
            gr.Checkbox(label="Randomize seed", value=True),
        ],
        outputs=[
            # Output components
            gr.Image(type="numpy", label="Generated Image"),
            gr.Textbox(label="Model Info"),
            gr.Number(label="Seed Used"),
        ],
        title=title,
        description=description,
    )

    # Launch the web interface
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
