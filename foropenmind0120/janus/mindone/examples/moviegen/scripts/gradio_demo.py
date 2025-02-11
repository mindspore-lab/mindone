import datetime
import glob
import logging
import os
import sys
import time
from typing import List, Tuple

import gradio as gr
import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import path_type

import mindspore as ms
from mindspore import amp, nn

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.append(mindone_lib_path)
sys.path.append(os.path.join(__dir__, ".."))

from mg.models.tae import TemporalAutoencoder
from mg.pipelines import InferPipeline
from mg.utils import init_model, to_numpy

from mindone.utils import init_train_env, set_logger
from mindone.visualize import save_videos

logger = logging.getLogger(__name__)


def prepare_captions(
    ul2_dir: str, metaclip_dir: str, byt5_dir: str, rank_id: int = 0, device_num: int = 1
) -> Tuple[List[str], List[str], List[str]]:
    """Prepare caption embeddings from specified directories"""
    ul2_emb = sorted(glob.glob(os.path.join(ul2_dir, "*.npz")))
    metaclip_emb = sorted(glob.glob(os.path.join(metaclip_dir, "*.npz")))
    byt5_emb = sorted(glob.glob(os.path.join(byt5_dir, "*.npz")))

    if len(ul2_emb) != len(byt5_emb):
        raise ValueError(
            f"ul2_dir ({len(ul2_emb)}), metaclip_dir ({len(metaclip_emb)}), "
            f" and byt5_dir ({len(byt5_emb)}) must contain the same number of files"
        )

    ul2_emb = ul2_emb[rank_id::device_num]
    logger.info(f"Number of captions for rank {rank_id}: {len(ul2_emb)}")
    return ul2_emb, metaclip_emb[rank_id::device_num], byt5_emb[rank_id::device_num]


def load_embeddings(selected_prompts: List[str], args) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
    """Load embeddings for selected prompts matching original implementation"""
    # Get full paths for selected prompts
    # print(selected_prompts)
    ul2_files = os.path.join(args.text_emb.ul2_dir, f"{selected_prompts}.npz")
    byt5_files = os.path.join(args.text_emb.byt5_dir, f"{selected_prompts}.npz")

    # Load embeddings in batch
    ul2_emb = ms.Tensor(np.load(ul2_files)["text_emb"], dtype=ms.float32)
    byt5_emb = ms.Tensor(np.load(byt5_files)["text_emb"], dtype=ms.float32)
    ul2_emb = ul2_emb.unsqueeze(0)
    byt5_emb = byt5_emb.unsqueeze(0)

    # Create placeholder metaclip embedding matching batch size
    metaclip_emb = ms.Tensor(np.ones((ul2_emb.shape[0], 300, 1280)), dtype=ms.float32)
    return ul2_emb, metaclip_emb, byt5_emb


def init_models(args):
    """Initialize MovieGen models with specified configurations"""
    # Initialize TAE
    logger.info("Initializing TAE...")
    tae = TemporalAutoencoder(**args.tae).set_train(False)
    if tae.dtype != ms.float32:
        amp.custom_mixed_precision(
            tae, black_list=amp.get_black_list() + [nn.GroupNorm, nn.AvgPool2d, nn.Upsample], dtype=tae.dtype
        )

    # Initialize Transformer model
    logger.info("Initializing Transformer model...")
    model = init_model(in_channels=tae.out_channels, **args.model).set_train(False)

    return model, tae


def create_pipeline(model, tae, args):
    """Create MovieGen inference pipeline"""
    img_h, img_w = args.image_size if isinstance(args.image_size, list) else (args.image_size, args.image_size)
    latent_size = tae.get_latent_size((args.num_frames, img_h, img_w))

    return InferPipeline(
        model,
        tae,
        latent_size,
        guidance_scale=args.guidance_scale,
        num_sampling_steps=args.num_sampling_steps,
        sample_method=args.sample_method,
        micro_batch_size=args.micro_batch_size,
    )


def generate_video(selected_prompts: List[str], args, pipeline, progress=gr.Progress()) -> List[str]:
    """Generate videos for selected prompts"""
    progress(0.1, "Loading embeddings...")
    ul2_emb, metaclip_emb, byt5_emb = load_embeddings(selected_prompts, args)

    progress(0.2, "Generating videos...")
    start_time = time.perf_counter()
    sample, latent = pipeline(
        ul2_emb=ul2_emb,
        metaclip_emb=metaclip_emb,
        byt5_emb=byt5_emb,
        num_frames=args.num_frames,
    )
    # import pdb
    # pdb.set_trace()
    generation_time = time.perf_counter() - start_time

    progress(0.8, "Saving videos...")
    save_dir = os.path.join(args.output_path, "gradio_samples")
    if args.append_timestamp:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir = os.path.join(save_dir, time_str)
    os.makedirs(save_dir, exist_ok=True)

    output_files = []
    # for i, prompt in enumerate(selected_prompts):
    output_file = os.path.join(save_dir, f"{selected_prompts}.{args.save_format}")
    save_videos(to_numpy(sample[0]), output_file, fps=args.fps)
    output_files.append(output_file)

    logger.info(
        f"Videos generated in {generation_time: .2f}s "
        f"({args.num_sampling_steps * len(selected_prompts) / generation_time: .2f} steps/s)"
    )

    return output_files


def create_demo(args):
    """Create and configure Gradio interface"""
    # Initialize models and pipeline
    model, tae = init_models(args)
    pipeline = create_pipeline(model, tae, args)

    # Get available prompts
    ul2_emb, _, _ = prepare_captions(**args.text_emb)
    prompts = [os.path.basename(p)[:-4] for p in ul2_emb]

    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# MovieGen Video Generation Demo")
        gr.Markdown(f"Model: {args.model.name}")

        with gr.Row():
            with gr.Column():
                prompt = gr.Dropdown(
                    choices=prompts,
                    label="Select Pre-computed Prompt",
                    info="Choose from available pre-computed prompts",
                )
                generate_btn = gr.Button("Generate Video", variant="primary")

            with gr.Column():
                video_output = gr.Video(label="Generated Video")
                info_box = gr.Textbox(label="Generation Info", interactive=False)

        def generate_and_log(prompt_name):
            print("Prompt name ", prompt_name)
            output_file = generate_video(prompt_name, args, pipeline)
            info = f"Successfully generated video for prompt: {prompt_name}"
            return output_file[0], info

        generate_btn.click(
            fn=generate_and_log,
            inputs=[prompt],
            outputs=[video_output, info_box],
        )

    return demo


if __name__ == "__main__":
    parser = ArgumentParser(description="MovieGen Gradio demo")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to MovieGen config file",
    )

    # Add all necessary arguments
    parser.add_function_arguments(init_train_env, "env")
    parser.add_function_arguments(init_model, "model", skip={"in_channels"})

    # TAE parameters
    tae_group = parser.add_argument_group("TAE parameters")
    tae_group.add_class_arguments(TemporalAutoencoder, "tae", instantiate=False)

    # Inference parameters
    infer_group = parser.add_argument_group("Inference parameters")
    infer_group.add_class_arguments(InferPipeline, skip={"model", "tae", "latent_size"}, instantiate=False)
    infer_group.add_argument("--image_size", type=int, nargs="+", default=[256, 455])
    infer_group.add_argument("--num_frames", type=int, default=32)
    infer_group.add_argument("--fps", type=int, default=16)
    infer_group.add_function_arguments(prepare_captions, "text_emb", skip={"rank_id", "device_num"})
    infer_group.add_argument("--batch_size", type=int, default=2)

    # Save options
    save_group = parser.add_argument_group("Saving options")
    save_group.add_argument("--save_format", default="mp4", choices=["gif", "mp4", "png"])
    save_group.add_argument("--output_path", default="output/", type=path_type("dcc"))
    save_group.add_argument("--append_timestamp", type=bool, default=True)
    save_group.add_argument(
        "--save_latent",
        type=bool,
        default=False,
        help="Save denoised video latent. If True, the denoised latents will be saved in $output_path/denoised_latents",
    )
    args = parser.parse_args()

    # Set up logging
    os.makedirs(os.path.join(args.output_path, "logs"), exist_ok=True)
    set_logger(name="", output_dir=os.path.join(args.output_path, "logs"))

    # Create and launch demo
    demo = create_demo(args)
    demo.launch()
