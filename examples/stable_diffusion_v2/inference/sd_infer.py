import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import mindspore as ms

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(workspace))
from conditions.canny.canny_detector import CannyDetector
from conditions.segmentation.segment_detector import SegmentDetector
from conditions.utils import HWC3, resize_image
from ldm.modules.logger import set_logger
from ldm.util import instantiate_from_config, str2bool
from libs.helper import VaeImageProcessor, load_model_from_config, set_env
from libs.sd_models import SDControlNet, SDImg2Img, SDInpaint, SDText2Img

logger = logging.getLogger("Stable Diffusion Inference")


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
        ckpt=config.model.pretrained_ckpt,
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
    batch_size = args.n_samples
    prompt = args.inputs.prompt
    if args.inputs.a_prompt:
        prompt = prompt + ", " + args.inputs.a_prompt
    data = batch_size * [prompt]
    negative_prompt = args.inputs.negative_prompt
    assert negative_prompt is not None
    negative_data = batch_size * [negative_prompt]
    # post-process negative prompts
    assert len(negative_data) <= len(data), "Negative prompts should be shorter than positive prompts"
    if len(negative_data) < len(data):
        logger.info("Negative prompts are shorter than positive prompts, padding blank prompts")
        blank_negative_prompt = batch_size * [""]
        for _ in range(len(data) - len(negative_data)):
            negative_data.append(blank_negative_prompt)

    # create inputs
    inputs = {}
    inputs["prompt"] = prompt
    inputs["prompt_data"] = model.tokenize(data)
    inputs["negative_prompt"] = negative_prompt
    inputs["negative_prompt_data"] = model.tokenize(negative_data)
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
    if args.task == "text2img":
        sd_infer = SDText2Img(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=model.scale_factor,
            num_inference_steps=args.sampling_steps,
        )
    elif args.task == "img2img":
        sd_infer = SDImg2Img(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=model.scale_factor,
            num_inference_steps=args.sampling_steps,
        )
        init_image = Image.open(args.inputs.image_path).convert("RGB")
        img = img_processor.preprocess(init_image, height=args.inputs.H, width=args.inputs.W)
        inputs["img"] = img.repeat(batch_size, axis=0)
        init_timestep = min(int(args.sampling_steps * args.inputs.strength), args.sampling_steps)
        t_start = max(args.sampling_steps - init_timestep, 0)
        inputs["timesteps"] = inputs["timesteps"][t_start * scheduler.order :]
    elif args.task == "inpaint":
        sd_infer = SDInpaint(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=model.scale_factor,
            num_inference_steps=args.sampling_steps,
        )
        init_image = Image.open(args.inputs.image_path).convert("RGB")
        image = img_processor.resize(init_image, args.inputs.H, args.inputs.W)
        image = np.array(image)[None].transpose(0, 3, 1, 2)
        image = (image / 127.5 - 1.0).astype(np.float32)
        mask = Image.open(args.inputs.mask_path).convert("L")
        mask = img_processor.resize(mask, args.inputs.H, args.inputs.W, resample="nearest")
        mask = (np.array(mask)[None, None] / 255.0 > 0.5).astype(np.float32)
        masked_image = image * (1 - mask)
        inputs["masked_image"] = ms.Tensor(np.repeat(masked_image, batch_size, axis=0), ms.float16)
        inputs["mask"] = ms.Tensor(np.repeat(mask, batch_size, axis=0), ms.float16)
    elif args.task == "controlnet":
        sd_infer = SDControlNet(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=model.scale_factor,
            num_inference_steps=args.sampling_steps,
        )

        image = cv2.imread(args.inputs.image_path)
        input_image = np.array(image, dtype=np.uint8)
        img = resize_image(HWC3(input_image), args.inputs.image_resolution)
        H, W, C = img.shape
        args.inputs.H = H
        args.inputs.W = W
        if args.controlnet_mode == "canny":
            apply_canny = CannyDetector()
            detected_map = apply_canny(img, args.inputs.low_threshold, args.inputs.high_threshold)
            detected_map = HWC3(detected_map)
        elif args.controlnet_mode == "segmentation":
            if os.path.exists(args.inputs.condition_ckpt_path):
                apply_segment = SegmentDetector(ckpt_path=args.inputs.condition_ckpt_path)
            else:
                logger.warning(
                    f"!!!Warning!!!: Condition Detector checkpoint path {args.inputs.condition_ckpt_path} doesn't exist"
                )
            detected_map = apply_segment(img)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            raise NotImplementedError(f"mode {args.controlnet_mode} not supported")

        Image.fromarray(detected_map).save(os.path.join(args.sample_path, "detected_map.png"))

        control = detected_map.copy().astype(np.float32) / 255.0
        control = np.transpose(control, (2, 0, 1))
        control = np.stack([control for _ in range(batch_size)], axis=0).astype(np.float16)
        inputs["control"] = ms.Tensor(control, ms.float16)

    else:
        raise ValueError(f"Not support task: {args.task}")

    logger.info(
        f"Generating images with conditions:\n"
        f"Prompt(s): {inputs['prompt']}, \n"
        f"Negative prompt(s): {inputs['negative_prompt']}"
    )

    for n in range(args.n_iter):
        start_time = time.time()
        noise = np.random.randn(args.n_samples, 4, args.inputs.H // 8, args.inputs.W // 8)
        inputs["noise"] = ms.Tensor(noise, ms.float16)

        x_samples = sd_infer(inputs)
        x_samples = img_processor.postprocess(x_samples)

        for sample in x_samples:
            sample.save(os.path.join(args.sample_path, f"{args.base_count:05}.png"))
            args.base_count += 1

        end_time = time.time()
        logger.info(
            "{}/{} images generated, time cost for current trial: {:.3f}s".format(
                batch_size * (n + 1), batch_size * args.n_iter, end_time - start_time
            )
        )

    logger.info(f"Done! All generated images are saved in: {args.output_path}/samples" f"\nEnjoy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument(
        "--device_target", type=str, default="Ascend", help="Device target, should be in [Ascend, GPU, CPU]"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text2img",
        help="Task name, should be [text2img, img2img, inpaint, controlnet], "
        "if choose a task name, use the config/[task].yaml for inputs",
        choices=["text2img", "img2img", "inpaint", "controlnet"],
    )
    parser.add_argument("--model", type=str, required=True, help="path to config which constructs model.")
    parser.add_argument("--output_path", type=str, default="output", help="dir to write results to")
    parser.add_argument("--sampler", type=str, default="./config/schedule/ddim.yaml", help="infer sampler yaml path")
    parser.add_argument("--sampling_steps", type=int, default=50, help="number of sampling steps")
    parser.add_argument("--n_iter", type=int, default=1, help="number of iterations or trials.")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
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
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--log_level", type=str, default="INFO", help="log level, options: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument(
        "--controlnet_mode",
        type=str,
        default="canny",
        help="control mode for controlnet, should be in [canny, segmentation]",
    )
    args = parser.parse_args()
    set_logger(name="", output_dir=args.output_path, rank=0, log_level=args.log_level)

    if not os.path.exists(args.model):
        raise ValueError(
            f"model config file {args.model} is not exist!, please set it by --model=xxx.yaml. "
            f"eg. --model=./config/model/v2-inference.yaml"
        )
    if not os.path.isabs(args.model):
        args.model = os.path.join(workspace, args.model)
    if args.task == "text2img":
        inputs_config_path = "./config/text2img.yaml"
        default_ckpt = "./models/sd_v2_base-57526ee4.ckpt"
    elif args.task == "img2img":
        inputs_config_path = "./config/img2img.yaml"
        default_ckpt = "./models/sd_v2_base-57526ee4.ckpt"
    elif args.task == "inpaint":
        inputs_config_path = "./config/inpaint.yaml"
        default_ckpt = "./models/sd_v2_inpaint-f694d5cf.ckpt"
    elif args.task == "controlnet":
        if args.controlnet_mode == "canny":
            inputs_config_path = "./config/controlnet_canny.yaml"
            default_ckpt = "./models/control_canny_sd_v1.5_static-6350d204.ckpt"
        elif args.controlnet_mode == "segmentation":
            inputs_config_path = "./config/controlnet_segmentation.yaml"
            default_ckpt = "./models/control_segmentation_sd_v1.5_static-77bea2e9.ckpt"
        else:
            raise NotImplementedError(f"mode {args.controlnet_mode} not supported")
    else:
        raise ValueError(f"{args.task} is invalid, should be in [text2img, img2img, inpaint]")
    inputs = OmegaConf.load(inputs_config_path)

    key_settings_info = ["Key Settings:\n" + "=" * 50]
    key_settings_info += [
        f"SD infer task: {args.task}",
        f"model config: {args.model}",
        f"inputs config: {inputs_config_path}",
        f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
        f"Number of trials for each prompt: {args.n_iter}",
        f"Number of samples in each trial: {args.n_samples}",
        f"Sampler: {args.sampler}",
        f"Sampling steps: {args.sampling_steps}",
        f"Uncondition guidance scale: {args.scale}",
    ]
    for key in inputs.keys():
        key_settings_info.append(f"{key}: {inputs[key]}")

    logger.info("\n".join(key_settings_info))

    args.inputs = inputs
    main(args)
