import argparse
import logging
import os
import sys

import numpy as np
from conditions.utils import create_video, pre_process_canny, prepare_video
from omegaconf import OmegaConf

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(workspace, "../stable_diffusion_v2/")))
from inference.libs.helper import VaeImageProcessor, load_model_from_config, set_env
from inference.libs.sd_models import SDControlNet
from ldm.modules.logger import set_logger
from ldm.util import instantiate_from_config, str2bool

logger = logging.getLogger("Text2Video-Zero Inference")


class NoisePrepare(nn.Cell):
    def __init__(self, frame, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        self.frame = frame

    def construct(self, noise):
        noise = ops.cast(noise, ms.float16)
        noise = ops.tile(noise, (self.frame, 1, 1, 1))
        return noise


def inference_text2video(
    control, inputs, noise, prompt_data, negative_prompt_data, sd_infer, img_processor, chunk_size=8
):
    frames_counter = 0
    f = control.shape[0]
    chunk_ids = np.arange(0, f, chunk_size - 1)
    result = []
    for i in range(len(chunk_ids)):
        ch_start = chunk_ids[i]
        ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
        frame_ids = [0] + list(range(ch_start, ch_end))
        print(f"Processing chunk {i + 1} / {len(chunk_ids)}")
        inputs["prompt_data"] = prompt_data[frame_ids]
        inputs["negative_prompt_data"] = negative_prompt_data[frame_ids]
        inputs["noise"] = noise[frame_ids]
        inputs["control"] = control[frame_ids]
        result_frame = sd_infer(inputs)[1:]
        result_frame = img_processor.postprocess(result_frame, output_type="np")
        result.append(result_frame)
        frames_counter += len(chunk_ids) - 1

    result = np.concatenate(result)
    return result


def main(args):
    # set logger
    set_env(args)
    ms.set_context(device_target=args.device_target)
    # create model
    config = OmegaConf.load(f"{args.model}")
    version = config.model.version
    os.environ["SD_VERSION"] = version
    model = load_model_from_config(
        config,
        ckpt=args.inputs.pretrained_ckpt,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_only_ckpt=args.lora_ckpt_path,
    )
    sampler_config = OmegaConf.load(args.sampler)
    if config.model.prediction_type == "v":
        sampler_config.params.prediction_type = "v_prediction"
    scheduler = instantiate_from_config(sampler_config)
    timesteps = scheduler.set_timesteps(args.sampling_steps)

    # read prompts
    prompt = args.inputs.prompt
    if "a_prompt" in args.inputs:
        prompt = prompt + ", " + args.inputs.a_prompt
    negative_prompt = args.inputs.negative_prompt
    assert negative_prompt is not None

    # create inputs
    inputs = {}
    inputs["prompt"] = prompt
    inputs["negative_prompt"] = negative_prompt
    inputs["timesteps"] = timesteps
    inputs["scale"] = ms.Tensor(args.scale, ms.float16)

    # create model
    text_encoder = model.cond_stage_model
    unet = model.model
    vae = model.first_stage_model
    img_processor = VaeImageProcessor()
    if args.device_target != "Ascend":
        unet.to_float(ms.float32)
        vae.to_float(ms.float32)

    sd_infer = SDControlNet(
        text_encoder,
        unet,
        vae,
        scheduler,
        scale_factor=model.scale_factor,
        num_inference_steps=args.sampling_steps,
    )
    video, fps = prepare_video(args.inputs.video_path, args.inputs.image_resolution, False)
    if args.inputs.controlnet_mode == "canny":
        control = pre_process_canny(video, args.inputs.low_threshold, args.inputs.high_threshold).astype(ms.float16)
    else:
        raise NotImplementedError(f"mode {args.inputs.controlnet_mode} not supported")
    f, h, w, _ = video.shape
    args.inputs.H = h
    args.inputs.W = w
    prompt_data = [prompt] * f
    negative_prompt_data = [negative_prompt] * f

    assert len(negative_prompt_data) <= len(prompt_data), "Negative prompts should be shorter than positive prompts"
    if len(negative_prompt_data) < len(prompt_data):
        logger.info("Negative prompts are shorter than positive prompts, padding blank prompts")
        blank_negative_prompt = f * [""]
        for _ in range(len(prompt_data) - len(negative_prompt_data)):
            negative_prompt_data.append(blank_negative_prompt)

    prompt_data = model.tokenize(prompt_data)
    negative_prompt_data = model.tokenize(negative_prompt_data)

    logger.info(
        f"Generating images with conditions:\n"
        f"Prompt(s): {inputs['prompt']}, \n"
        f"Negative prompt(s): {inputs['negative_prompt']}"
    )

    noise_prepare = NoisePrepare(frame=f)
    noise = ops.standard_normal((1, 4, args.inputs.H // 8, args.inputs.W // 8))
    noise = noise_prepare(noise)

    result = inference_text2video(control, inputs, noise, prompt_data, negative_prompt_data, sd_infer, img_processor)
    video_name = args.inputs.video_path.split("/")[-1]
    video_name = args.inputs.controlnet_mode + "_" + video_name
    save_path = os.path.join(args.sample_path, video_name)
    create_video(result, fps, path=save_path)

    logger.info(f"Done! Generated video is saved in: {args.output_path}/samples" f"\nEnjoy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument(
        "--device_target", type=str, default="Ascend", help="Device target, should be in [Ascend, GPU, CPU]"
    )
    parser.add_argument("--model", type=str, required=True, help="path to config which constructs model.")
    parser.add_argument("--output_path", type=str, default="output", help="dir to write results to")
    parser.add_argument("--sampler", type=str, default="./config/schedule/ddim.yaml", help="infer sampler yaml path")
    parser.add_argument("--sampling_steps", type=int, default=50, help="number of sampling steps")
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: "
        "eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
        "Simplified: `uc + scale * (uc - prompt)`",
    )
    parser.add_argument(
        "--use_lora",
        default=False,
        type=str2bool,
        help="whether the checkpoint used for inference is finetuned from LoRA",
    )
    parser.add_argument(
        "--lora_rank",
        default=None,
        type=int,
        help="LoRA rank. If None, lora checkpoint should contain the value for lora rank in its append_dict.",
    )
    parser.add_argument(
        "--lora_ckpt_path", type=str, default=None, help="path to lora only checkpoint. Set it if use_lora is not None"
    )
    parser.add_argument("--seed", type=int, default=41, help="the seed (for reproducible sampling)")
    parser.add_argument("--log_level", type=str, default="INFO", help="log level, options: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument(
        "--inputs_config_path",
        type=str,
        default="./config/text2video-zero.yaml",
        help="the path for config file including necessary input arguments",
    )
    args = parser.parse_args()
    set_logger(name="", output_dir=args.output_path, rank=0, log_level=args.log_level)

    if not os.path.exists(args.model):
        raise ValueError(
            f"model config file {args.model} is not exist!, please set it by --model=xxx.yaml. "
            f"eg. --model=./config/model/v1-inference-text2video-zero.yaml"
        )
    if not os.path.isabs(args.model):
        args.model = os.path.join(workspace, args.model)
    inputs = OmegaConf.load(args.inputs_config_path)

    key_settings_info = ["Key Settings:\n" + "=" * 50]
    key_settings_info += [
        f"model config: {args.model}",
        f"inputs config: {args.inputs_config_path}",
        f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
        f"Sampler: {args.sampler}",
        f"Sampling steps: {args.sampling_steps}",
        f"Uncondition guidance scale: {args.scale}",
    ]
    for key in inputs.keys():
        key_settings_info.append(f"{key}: {inputs[key]}")

    logger.info("\n".join(key_settings_info))

    args.inputs = inputs
    main(args)
