import argparse
import logging
import os
import sys
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import mindspore as ms

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(workspace))

from ldm.models.clip.simple_tokenizer import get_tokenizer
from ldm.modules.logger import set_logger
from ldm.util import instantiate_from_config
from libs.helper import VaeImageProcessor
from libs.infer_engine.sd_lite_models import SDLiteImg2Img, SDLiteInpaint, SDLiteText2Img

logger = logging.getLogger("Stable Diffusion Lite Deploy")


def tokenize(tokenizer, texts):
    SOT_TEXT = tokenizer.sot_text
    EOT_TEXT = tokenizer.eot_text
    CONTEXT_LEN = 77

    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder[SOT_TEXT]
    eot_token = tokenizer.encoder[EOT_TEXT]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), CONTEXT_LEN), np.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]

        result[i, : len(tokens)] = np.array(tokens, np.int64)

    return result.astype(np.int32)


def get_mindir_path(model_save_path, name):
    mindir_path = os.path.join(model_save_path, f"{name}_lite.mindir")
    if not os.path.exists(mindir_path):
        mindir_path = os.path.join(model_save_path, f"{name}_lite_graph.mindir")
    return mindir_path


def main(args):
    # set logger
    set_logger(name="", output_dir=args.output_path, rank=0, log_level=args.log_level)
    ms.set_context(device_target="CPU")
    args.sample_path = os.path.join(args.output_path, "samples")
    args.base_count = len(os.listdir(args.sample_path))

    model_config = OmegaConf.load(args.model)
    version = model_config.model.version
    os.environ["SD_VERSION"] = version
    sampler_config = OmegaConf.load(args.sampler)
    if model_config.model.prediction_type == "v":
        sampler_config.params.prediction_type = "v_prediction"
    scheduler = instantiate_from_config(sampler_config)
    timesteps = scheduler.set_timesteps(args.sampling_steps)
    scheduler_type = sampler_config.type
    img_processor = VaeImageProcessor()
    args.model_save_path = f"{model_config.model.name}-{args.task}"
    tokenizer = get_tokenizer(model_config.model.params.cond_stage_config.params.tokenizer_name)
    model_save_path = os.path.join(args.output_path, args.model_save_path)
    logger.info(f"model_save_path: {model_save_path}")

    data_prepare = get_mindir_path(model_save_path, args.inputs.data_prepare_model)
    scheduler_preprocess = get_mindir_path(model_save_path, f"{args.inputs.scheduler_preprocess}-{scheduler_type}")
    predict_noise = get_mindir_path(model_save_path, args.inputs.predict_noise_model)
    noisy_sample = get_mindir_path(model_save_path, f"{args.inputs.noisy_sample_model}-{scheduler_type}")
    vae_decoder = get_mindir_path(model_save_path, args.inputs.vae_decoder_model)

    # read prompts
    batch_size = args.n_samples
    prompt = args.inputs.prompt
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
    inputs["prompt_data"] = tokenize(tokenizer, data)
    inputs["negative_prompt"] = negative_prompt
    inputs["negative_prompt_data"] = tokenize(tokenizer, negative_data)
    inputs["timesteps"] = timesteps
    inputs["scale"] = np.array(args.scale, np.float16)

    # create model
    if args.task == "text2img":
        sd_infer = SDLiteText2Img(
            data_prepare,
            scheduler_preprocess,
            predict_noise,
            noisy_sample,
            vae_decoder,
            device_target=args.device_target,
            device_id=int(os.getenv("DEVICE_ID", 0)),
            num_inference_steps=args.sampling_steps,
        )
    elif args.task == "img2img":
        sd_infer = SDLiteImg2Img(
            data_prepare,
            scheduler_preprocess,
            predict_noise,
            noisy_sample,
            vae_decoder,
            device_target=args.device_target,
            device_id=int(os.getenv("DEVICE_ID", 0)),
            num_inference_steps=args.sampling_steps,
        )
        init_image = Image.open(args.inputs.image_path).convert("RGB")
        img = img_processor.preprocess(init_image, height=args.inputs.H, width=args.inputs.W)
        inputs["img"] = img.repeat(batch_size, axis=0).asnumpy()
        init_timestep = min(int(args.sampling_steps * args.inputs.strength), args.sampling_steps)
        t_start = max(args.sampling_steps - init_timestep, 0)
        inputs["timesteps"] = inputs["timesteps"][t_start * scheduler.order :]
    elif args.task == "inpaint":
        sd_infer = SDLiteInpaint(
            data_prepare,
            scheduler_preprocess,
            predict_noise,
            noisy_sample,
            vae_decoder,
            device_target=args.device_target,
            device_id=int(os.getenv("DEVICE_ID", 0)),
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
        inputs["masked_image"] = np.repeat(masked_image, batch_size, axis=0).astype(np.float16)
        inputs["mask"] = np.repeat(mask, batch_size, axis=0).astype(np.float16)
    else:
        raise ValueError(f"Not support task: {args.task}")

    for n in range(args.n_iter):
        start_time = time.time()
        inputs["noise"] = np.random.standard_normal(
            size=(batch_size, 4, args.inputs.H // 8, args.inputs.W // 8)
        ).astype(np.float16)

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
        "--device_target",
        type=str,
        default="Ascend",
        help="Device target, should be in [Ascend]",
        choices=["Ascend"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text2img",
        help="Task name, should be [text2img, img2img], "
        "if choose a task name, use the config/[task].yaml for inputs",
        choices=["text2img", "img2img", "inpaint"],
    )
    parser.add_argument("--model", type=str, default=None, help="path to config which constructs model.")
    parser.add_argument("--output_path", type=str, default="output", help="dir to write results to")
    parser.add_argument("--sampler", type=str, default="config/schedule/ddim.yaml", help="infer sampler yaml path")
    parser.add_argument("--sampling_steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--n_iter", type=int, default=1, help="number of iterations or trials. sample this often, ")
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
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--log_level", type=str, default="INFO", help="log level, options: DEBUG, INFO, WARNING, ERROR")
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
    elif args.task == "img2img":
        inputs_config_path = "./config/img2img.yaml"
    elif args.task == "inpaint":
        inputs_config_path = "./config/inpaint.yaml"
    else:
        raise ValueError(f"{args.task} is invalid, should be in [text2img, img2img, inpaint]")
    inputs = OmegaConf.load(inputs_config_path)

    key_settings_info = ["Key Settings:\n" + "=" * 50]
    key_settings_info += [
        f"SD Lite infer task: {args.task}",
        f"model config: {args.model}",
        f"inputs config: {inputs_config_path}",
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
