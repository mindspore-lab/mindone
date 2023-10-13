import argparse
import logging
import os
import sys
import time

import numpy as np
from libs.tokenizer import get_tokenizer
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import ops

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(workspace))
from gm.util import instantiate_from_config
from libs.helper import VaeImageProcessor, load_model_from_config, set_env
from libs.logger import set_logger
from libs.sd_models import SDText2Img
from transformers import CLIPTokenizer

logger = logging.getLogger("Stable Diffusion XL Inference")


def tokenize(batch, emb_models_config):
    tokens = []
    for c in emb_models_config:
        tokenizer = c["tokenizer"]
        if isinstance(tokenizer, CLIPTokenizer):
            token = tokenizer(
                batch[c["input_key"]],
                truncation=True,
                max_length=77,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
            )["input_ids"]
        else:
            token = tokenizer(batch[c["input_key"]])
        if isinstance(token, list):
            token = np.array(token)
        tokens.append(token)
    return tokens


def main(args):
    # set logger
    set_env(args)
    ms.set_context(device_target=args.device_target)

    # create sampler
    sampler_config = OmegaConf.load(args.sampler)
    scheduler = instantiate_from_config(sampler_config)

    # create model
    model_config = OmegaConf.load(f"{args.model}")
    model = load_model_from_config(
        model_config.model,
        ckpt=model_config.model.pretrained_ckpt,
        freeze=True,
        load_filter=False,
        amp_level=args.ms_amp_level,
    )
    emb_models = model_config.model.params.conditioner_config.params.emb_models
    emb_models_config = []
    for emb_model in emb_models:
        emb_model_config = {"input_key": emb_model.input_key, "tokenizer": get_tokenizer(emb_model.tokenizer_name)}
        emb_models_config.append(emb_model_config)

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
    batch = {
        "txt": data,
        "crop_coords_top_left": [[0.0, 0.0]],
        "target_size_as_tuple": [[1024.0, 1024.0]],
        "original_size_as_tuple": [[1024.0, 1024.0]],
    }
    batch_uc = {
        "txt": negative_data,
        "crop_coords_top_left": [[0.0, 0.0]],
        "target_size_as_tuple": [[1024.0, 1024.0]],
        "original_size_as_tuple": [[1024.0, 1024.0]],
    }

    inputs = {}
    batch_token = tokenize(batch, emb_models_config)
    batch_uc_token = tokenize(batch_uc, emb_models_config)

    pos_clip_token = np.concatenate((batch_token[0], batch_token[1]), axis=0).astype(np.int32)
    pos_time_token = np.concatenate((batch_token[2], batch_token[3], batch_token[4]), axis=0).astype(np.float16)

    neg_clip_token = np.concatenate((batch_uc_token[0], batch_uc_token[1]), axis=0).astype(np.int32)
    neg_time_token = np.concatenate((batch_uc_token[2], batch_uc_token[3], batch_uc_token[4]), axis=0).astype(
        np.float16
    )

    inputs["pos_clip_token"] = pos_clip_token
    inputs["pos_time_token"] = pos_time_token
    inputs["neg_clip_token"] = neg_clip_token
    inputs["neg_time_token"] = neg_time_token
    inputs["timesteps"] = args.sampling_steps
    inputs["scale"] = ms.Tensor(args.scale, ms.float32)

    # create model
    img_processor = VaeImageProcessor()
    if args.task == "text2img":
        sd_infer = SDText2Img(
            model.conditioner,
            model.model,
            model.first_stage_model,
            scheduler,
            model.denoiser,
            scale_factor=model.scale_factor,
            num_inference_steps=args.sampling_steps,
        )
    else:
        raise ValueError(f"Not support task: {args.task}")

    logger.info(
        f"Generating images with conditions:\n" f"Prompt(s): {prompt}, \n" f"Negative prompt(s): {negative_prompt}"
    )

    for n in range(args.n_iter):
        start_time = time.time()
        inputs["noise"] = ops.standard_normal((args.n_samples, 4, args.inputs.H // 8, args.inputs.W // 8)).astype(
            ms.float32
        )
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
        help="Task name, should be [text2img, img2img, inpaint], "
        "if choose a task name, use the config/[task].yaml for inputs",
        choices=["text2img", "img2img", "inpaint"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./config/model/sd_xl_base_inference.yaml",
        help="path to config which constructs model.",
    )
    parser.add_argument("--output_path", type=str, default="output", help="dir to write results to")
    parser.add_argument(
        "--sampler", type=str, default="./config/schedule/euler_edm.yaml", help="infer sampler yaml path"
    )
    parser.add_argument("--sampling_steps", type=int, default=40, help="number of sampling steps")
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
        default=5.0,
        help="unconditional guidance scale: "
        "eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
        "Simplified: `uc + scale * (uc - prompt)`",
    )
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--log_level", type=str, default="INFO", help="log level, options: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--ms_amp_level", type=str, default="O2", help="mixed precision level")
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
