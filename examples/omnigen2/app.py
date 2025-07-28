import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Literal, Optional, Union

import gradio as gr
import numpy as np
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr, path_type
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2 import OmniGen2ChatPipeline, OmniGen2Pipeline
from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from omnigen2.utils.img_util import create_collage
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLProcessor

from mindspore import dtype
from mindspore.nn import no_init_parameters

from mindone.diffusers import AutoencoderKL
from mindone.transformers import Qwen2_5_VLForConditionalGeneration

NEGATIVE_PROMPT = (
    "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated,"
    " (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

pipeline = None
save_images = False

Path_u = path_type("u")  # URL to an image


@dataclass
class Example:
    prompt: str
    width: int
    height: int
    scheduler: Literal["dpmsolver++", "euler"]
    num_inference_steps: int
    text_guidance_scale: float
    image_guidance_scale: float
    cfg_range_start: float
    cfg_range_end: float
    num_images_per_prompt: int
    max_input_image_size: int
    max_image_size: int
    seed: int
    input_image: Optional[Union[Path_fr, Path_u]] = None
    input_image2: Optional[Union[Path_fr, Path_u]] = None
    input_image3: Optional[Union[Path_fr, Path_u]] = None
    negative_prompt: Optional[str] = None


@no_init_parameters()
def load_pipeline(weight_dtype, args) -> Union[OmniGen2Pipeline, OmniGen2ChatPipeline]:
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, subfolder="mllm", mindspore_dtype=weight_dtype
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path, subfolder="mllm_processor")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.model_path, subfolder="transformer", mindspore_dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae", mindspore_dtype=weight_dtype)
    if args.chat_mode:
        return OmniGen2ChatPipeline(transformer, vae, scheduler, mllm, processor)
    return OmniGen2Pipeline(transformer, vae, scheduler, mllm, processor)


def run(
    instruction,
    width_input,
    height_input,
    scheduler,
    num_inference_steps,
    image_input_1,
    image_input_2,
    image_input_3,
    negative_prompt,
    guidance_scale_input,
    img_guidance_scale_input,
    cfg_range_start,
    cfg_range_end,
    num_images_per_prompt,
    max_input_image_side_length,
    max_pixels,
    seed_input,
    progress=gr.Progress(),
):
    input_images = [image_input_1, image_input_2, image_input_3]
    input_images = [img for img in input_images if img is not None]

    if len(input_images) == 0:
        input_images = None

    if seed_input == -1:
        seed_input = random.randint(0, 2**16 - 1)

    generator = np.random.default_rng(seed_input)

    def progress_callback(cur_step, timesteps):
        frac = (cur_step + 1) / float(timesteps)
        progress(frac)

    if scheduler == "dpmsolver++":
        pipeline.scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++", solver_type="midpoint", solver_order=2, prediction_type="flow_prediction"
        )
    else:
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=width_input,
        height=height_input,
        max_input_image_side_length=max_input_image_side_length,
        max_pixels=max_pixels,
        num_inference_steps=num_inference_steps,
        max_sequence_length=1024,
        text_guidance_scale=guidance_scale_input,
        image_guidance_scale=img_guidance_scale_input,
        cfg_range=(cfg_range_start, cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        output_type="pil",
        step_func=progress_callback,
    )

    progress(1.0)

    if results.text.startswith("<|img|>"):
        vis_images = [np.array(image, dtype=np.float32) / 127.5 - 1 for image in results.images]
        output_image = create_collage(vis_images)

        if save_images:
            # Create outputs directory if it doesn't exist
            output_dir = os.path.join(ROOT_DIR, "outputs_gradio")
            os.makedirs(output_dir, exist_ok=True)

            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

            # Generate unique filename with timestamp
            output_path = os.path.join(output_dir, f"{timestamp}.png")
            # Save the image
            output_image.save(output_path)

            # Save All Generated Images
            if len(results.images) > 1:
                for i, image in enumerate(results.images):
                    image_name, ext = os.path.splitext(output_path)
                    image.save(f"{image_name}_{i}{ext}")
        return output_image, None
    else:
        return None, results.text


def get_examples(examples: list[Example]) -> list[list]:
    # Convert loaded examples to the expected format
    def _parse(example: Example) -> list:
        for field in ["input_image", "input_image2", "input_image3"]:
            if (img := example.__getattribute__(field)) is not None:
                if img.relative == "None":  # FIXME: jsonargparse bug
                    example.__setattr__(field, None)
                elif isinstance(img, Path_fr):
                    example.__setattr__(field, os.path.abspath(img))
                elif isinstance(img, Path_u):
                    example.__setattr__(field, Image.open(BytesIO(img.get_content())))
                else:
                    raise ValueError(f"Invalid image type {type(img)}")

        if example.negative_prompt is None:
            example.negative_prompt = NEGATIVE_PROMPT

        return [  # ensure correct order. TODO: simplify
            example.prompt,
            example.width,
            example.height,
            example.scheduler,
            example.num_inference_steps,
            example.input_image,
            example.input_image2,
            example.input_image3,
            example.negative_prompt,
            example.text_guidance_scale,
            example.image_guidance_scale,
            example.cfg_range_start,
            example.cfg_range_end,
            example.num_images_per_prompt,
            example.max_input_image_size,
            example.max_image_size,
            example.seed,
        ]

    with ThreadPoolExecutor(max_workers=len(examples)) as executor:
        cases = list(tqdm(executor.map(_parse, examples), total=len(examples), desc="Loading examples"))

    return cases


def run_for_examples(
    instruction,
    width_input,
    height_input,
    scheduler,
    num_inference_steps,
    image_input_1,
    image_input_2,
    image_input_3,
    negative_prompt,
    text_guidance_scale_input,
    image_guidance_scale_input,
    cfg_range_start,
    cfg_range_end,
    num_images_per_prompt,
    max_input_image_side_length,
    max_pixels,
    seed_input,
):
    return run(
        instruction,
        width_input,
        height_input,
        scheduler,
        num_inference_steps,
        image_input_1,
        image_input_2,
        image_input_3,
        negative_prompt,
        text_guidance_scale_input,
        image_guidance_scale_input,
        cfg_range_start,
        cfg_range_end,
        num_images_per_prompt,
        max_input_image_side_length,
        max_pixels,
        seed_input,
    )


description = """
### 💡 Quick Tips for Best Results (see our [github](https://github.com/VectorSpaceLab/OmniGen2?tab=readme-ov-file#-usage-tips) for more details)
- Image Quality: Use high-resolution images (**at least 512x512 recommended**).
- Be Specific: Instead of "Add bird to desk", try "Add the bird from image 1 to the desk in image 2".
- Use English: English prompts currently yield better results.
- Increase image_guidance_scale for better consistency with the reference image:
    - Image Editing: 1.3 - 2.0
    - In-context Generation: 2.0 - 3.0
- For in-context edit (edit based multiple images), we recommend using the following prompt format:
  "Edit the first image: add/replace (the [object] with) the [object] from the second image. [descripton for your target image]."
For example: "Edit the first image: add the man from the second image. The man is talking with a woman in the kitchen"

Compared to OmniGen 1.0, although OmniGen2 has made some improvements, some issues still remain. It may take multiple attempts to achieve a satisfactory result.
"""

article = """
```bibtex
@article{wu2025omnigen2,
  title={OmniGen2: Exploration to Advanced Multimodal Generation},
  author={Chenyuan Wu and Pengfei Zheng and Ruiran Yan and Shitao Xiao and Xin Luo and Yueze Wang and Wanli Li and
   Xiyan Jiang and Yexin Liu and Junjie Zhou and Ze Liu and Ziyi Xia and Chaofan Li and Haoge Deng and Jiahao Wang and
   Kun Luo and Bo Zhang and Defu Lian and Xinlong Wang and Zhongyuan Wang and Tiejun Huang and Zheng Liu},
  journal={arXiv preprint arXiv:2506.18871},
  year={2025}
}
```
"""


def main(args):
    # Gradio
    with gr.Blocks() as demo:
        gr.Markdown(
            "# OmniGen2: Exploration to Advanced Multimodal Generation [paper](https://arxiv.org/abs/2506.18871)"
            " [code](https://github.com/VectorSpaceLab/OmniGen2)"
        )
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                # text prompt
                instruction = gr.Textbox(
                    label='Enter your prompt. Use "first/second image" or “第一张图/第二张图” as reference.',
                    placeholder="Type your prompt here...",
                )

                with gr.Row(equal_height=True):
                    # input images
                    image_input_1 = gr.Image(label="First Image", type="pil")
                    image_input_2 = gr.Image(label="Second Image", type="pil")
                    image_input_3 = gr.Image(label="Third Image", type="pil")

                generate_button = gr.Button("Generate")

                negative_prompt = gr.Textbox(
                    label="Enter your negative prompt",
                    placeholder="Type your negative prompt here...",
                    value=NEGATIVE_PROMPT,
                )

                # slider
                with gr.Row(equal_height=True):
                    height_input = gr.Slider(label="Height", minimum=256, maximum=2048, value=1024, step=128)
                    width_input = gr.Slider(label="Width", minimum=256, maximum=2048, value=1024, step=128)
                with gr.Row(equal_height=True):
                    text_guidance_scale_input = gr.Slider(
                        label="Text Guidance Scale",
                        minimum=1.0,
                        maximum=8.0,
                        value=5.0,
                        step=0.1,
                    )

                    image_guidance_scale_input = gr.Slider(
                        label="Image Guidance Scale",
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                    )
                with gr.Row(equal_height=True):
                    cfg_range_start = gr.Slider(
                        label="CFG Range Start",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.1,
                    )

                    cfg_range_end = gr.Slider(
                        label="CFG Range End",
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.1,
                    )

                def adjust_end_slider(start_val, end_val):
                    return max(start_val, end_val)

                def adjust_start_slider(end_val, start_val):
                    return min(end_val, start_val)

                cfg_range_start.input(
                    fn=adjust_end_slider, inputs=[cfg_range_start, cfg_range_end], outputs=[cfg_range_end]
                )

                cfg_range_end.input(
                    fn=adjust_start_slider, inputs=[cfg_range_end, cfg_range_start], outputs=[cfg_range_start]
                )

                with gr.Row(equal_height=True):
                    scheduler_input = gr.Dropdown(
                        label="Scheduler",
                        choices=["euler", "dpmsolver++"],
                        value="euler",
                        info="The scheduler to use for the model.",
                    )

                    num_inference_steps = gr.Slider(label="Inference Steps", minimum=20, maximum=100, value=50, step=1)
                with gr.Row(equal_height=True):
                    num_images_per_prompt = gr.Slider(
                        label="Number of images per prompt",
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1,
                    )

                    seed_input = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, value=0, step=1)
                with gr.Row(equal_height=True):
                    max_input_image_side_length = gr.Slider(
                        label="max_input_image_side_length",
                        minimum=256,
                        maximum=2048,
                        value=2048,
                        step=256,
                    )
                    max_pixels = gr.Slider(
                        label="max_pixels",
                        minimum=256 * 256,
                        maximum=1536 * 1536,
                        value=1024 * 1024,
                        step=256 * 256,
                    )

            with gr.Column():
                with gr.Column():
                    # output image
                    output_image = gr.Image(label="Output Image")
                    global save_images
                    save_images = gr.Checkbox(label="Save generated images", value=False)
                    output_text = gr.Textbox(
                        label="Model Response",
                        placeholder="Text responses will appear here...",
                        lines=5,
                        visible=True,
                        interactive=False,
                    )

        global pipeline

        bf16 = True
        weight_dtype = dtype.bfloat16 if bf16 else dtype.float32

        pipeline = load_pipeline(weight_dtype, args)

        # click
        generate_button.click(
            run,
            inputs=[
                instruction,
                width_input,
                height_input,
                scheduler_input,
                num_inference_steps,
                image_input_1,
                image_input_2,
                image_input_3,
                negative_prompt,
                text_guidance_scale_input,
                image_guidance_scale_input,
                cfg_range_start,
                cfg_range_end,
                num_images_per_prompt,
                max_input_image_side_length,
                max_pixels,
                seed_input,
            ],
            outputs=[output_image, output_text],
        )

        gr.Examples(
            examples=get_examples(args.examples),
            fn=run_for_examples,
            inputs=[
                instruction,
                width_input,
                height_input,
                scheduler_input,
                num_inference_steps,
                image_input_1,
                image_input_2,
                image_input_3,
                negative_prompt,
                text_guidance_scale_input,
                image_guidance_scale_input,
                cfg_range_start,
                cfg_range_end,
                num_images_per_prompt,
                max_input_image_side_length,
                max_pixels,
                seed_input,
            ],
            outputs=[output_image, output_text],
        )

        gr.Markdown(article)
    # launch
    demo.launch(share=args.share, server_port=args.port, allowed_paths=[ROOT_DIR])


def parse_args():
    parser = ArgumentParser(description="Run the OmniGen2", default_config_files=["configs/app/examples.yaml"])
    parser.add_argument("--share", action="store_true", help="Share the Gradio app")
    parser.add_argument("--port", type=int, default=7860, help="Port to use for the Gradio app")
    parser.add_argument(
        "--model_path", type=str, default="OmniGen2/OmniGen2", help="Path or HuggingFace name of the model to load."
    )
    parser.add_argument("--chat_mode", action="store_true", help="Enable chat mode.")
    parser.add_function_arguments(get_examples)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
