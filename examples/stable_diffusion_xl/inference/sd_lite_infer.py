import argparse
import logging
import os
import sys
import time

import numpy as np
from libs.logger import set_logger
from libs.tokenizer import get_tokenizer
from omegaconf import OmegaConf
from transformers import CLIPTokenizer

import mindspore as ms

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(workspace))

from libs.helper import VaeImageProcessor
from libs.infer_engine.sd_lite_models import SDLiteText2Img

logger = logging.getLogger("Stable Diffusion XL Lite Deploy")


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


def get_mindir_path(model_save_path, name):
    mindir_path = os.path.join(model_save_path, f"{name}_lite.mindir")
    if not os.path.exists(mindir_path):
        mindir_path = os.path.join(model_save_path, f"{name}_graph_lite_graph.mindir")
    return os.path.abspath(mindir_path)


def main(args):
    # set logger
    set_logger(name="", output_dir=args.output_path, rank=0, log_level=args.log_level)
    ms.set_context(device_target="CPU")
    args.sample_path = os.path.join(args.output_path, "samples")
    args.base_count = len(os.listdir(args.sample_path))

    model_config = OmegaConf.load(args.model)
    sampler_config = OmegaConf.load(args.sampler)
    scheduler_type = sampler_config.type
    img_processor = VaeImageProcessor()
    args.model_save_path = f"{model_config.model.name}-{args.task}"

    emb_models = model_config.model.params.conditioner_config.params.emb_models
    emb_models_config = []
    for emb_model in emb_models:
        emb_model_config = {"input_key": emb_model.input_key, "tokenizer": get_tokenizer(emb_model.tokenizer_name)}
        emb_models_config.append(emb_model_config)

    model_save_path = os.path.join(args.output_path, args.model_save_path)
    logger.info(f"model_save_path: {model_save_path}")

    data_prepare = get_mindir_path(model_save_path, args.inputs.data_prepare_model)
    scheduler_preprocess = get_mindir_path(model_save_path, f"{args.inputs.scheduler_preprocess}-{scheduler_type}")
    scheduler_prepare_sampling_loop = get_mindir_path(
        model_save_path, f"{args.inputs.prepare_sampling_loop}-{scheduler_type}"
    )
    predict_noise = get_mindir_path(model_save_path, args.inputs.predict_noise_model)
    noisy_sample = get_mindir_path(model_save_path, f"{args.inputs.noisy_sample_model}-{scheduler_type}")
    vae_decoder = get_mindir_path(model_save_path, args.inputs.vae_decoder_model)
    denoiser = get_mindir_path(model_save_path, args.inputs.denoiser)

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
    # create inputs
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
    inputs["scale"] = np.array(args.scale, np.float16)

    # create model
    if args.task == "text2img":
        sd_infer = SDLiteText2Img(
            data_prepare,
            scheduler_preprocess,
            predict_noise,
            noisy_sample,
            vae_decoder,
            denoiser,
            scheduler_prepare_sampling_loop,
            device_target=args.device_target,
            device_id=int(os.getenv("DEVICE_ID", 0)),
            num_inference_steps=args.sampling_steps,
        )
    else:
        raise ValueError(f"Not support task: {args.task}")

    for n in range(args.n_iter):
        start_time = time.time()
        inputs["noise"] = np.random.standard_normal(
            size=(batch_size, 4, args.inputs.H // 8, args.inputs.W // 8)
        ).astype(np.float32)

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
    parser.add_argument(
        "--model",
        type=str,
        default="./config/model/sd_xl_base_inference.yaml",
        help="path to config which constructs model.",
    )
    parser.add_argument("--output_path", type=str, default="output", help="dir to write results to")
    parser.add_argument("--sampler", type=str, default="config/schedule/euler_edm.yaml", help="infer sampler yaml path")
    parser.add_argument("--sampling_steps", type=int, default=40, help="number of ddim sampling steps")
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
