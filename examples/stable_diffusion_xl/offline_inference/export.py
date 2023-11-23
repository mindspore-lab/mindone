import argparse
import logging
import os
import sys

from libs.helper import load_model_from_config, set_env
from libs.logger import set_logger
from libs.util import instantiate_from_config, str2bool
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import ops

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(workspace))
from libs.infer_engine.export_modules import (
    Denoiser,
    NoisySample,
    PredictNoise,
    SchedulerPreModelInput,
    SchedulerPrepareSamplingLoop,
    Text2ImgEmbedder,
    VAEDecoder,
)

logger = logging.getLogger("Stable Diffusion XL Export")


def model_export(net, inputs, name, model_save_path):
    if len(inputs) == 0:
        ms.export(net, file_name=os.path.join(model_save_path, name), file_format="MINDIR")
    elif len(inputs) == 1:
        ms.export(net, inputs, file_name=os.path.join(model_save_path, name), file_format="MINDIR")
    else:
        ms.export(net, *inputs, file_name=os.path.join(model_save_path, name), file_format="MINDIR")
    logger.info(f"convert {name} mindir done")


def main(args):
    # set logger
    set_env(args)
    ms.set_context(device_target="CPU")

    config = OmegaConf.load(f"{args.model}")
    # create sampler
    sampler_config = OmegaConf.load(args.sampler)
    scheduler = instantiate_from_config(sampler_config)
    scheduler_type = sampler_config.type

    args.model_save_path = f"{config.model.name}-{args.task}"
    model_save_path = os.path.join(args.output_path, args.model_save_path)
    os.makedirs(model_save_path, exist_ok=True)
    logger.info(f"model_save_path: {model_save_path}")
    converter = None
    if args.converte_lite:
        import mindspore_lite as mslite

        optimize_dict = {"ascend": "ascend_oriented", "gpu": "gpu_oriented", "cpu": "general"}
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = optimize_dict[args.device_target.lower()]
    if not args.only_converte_lite:
        # create model
        version = config.model.version
        os.environ["SD_VERSION"] = version
        model = load_model_from_config(
            config.model, ckpt=config.model.pretrained_ckpt, freeze=True, load_filter=False, amp_level=args.ms_amp_level
        )

        # data
        batch_size = args.n_samples
        clip_tokens = ops.ones((batch_size * 2, 77), ms.int32)
        time_tokens = ops.ones((batch_size * 3, 2), dtype=ms.float16)

        output_dim = 1024
        noise = ops.ones((batch_size, 4, args.inputs.H // 8, args.inputs.W // 8), ms.float32)
        ts = ops.ones((), ms.int32)
        scale = ops.ones((), ms.float16)
        sigma_hat_s = ops.ones(2, ms.float32)
        noised_input = ops.ones((batch_size * 2, 4, args.inputs.H // 8, args.inputs.W // 8), ms.float32)
        c_noise = ops.ones((2), ms.int32)
        y = ops.ones((batch_size * 2, 2816), ms.float32)
        context = ops.ones((batch_size * 2, 77, output_dim * 2), ms.float32)

        model_output = ops.ones((batch_size * 2, 4, args.inputs.H // 8, args.inputs.W // 8), ms.float32)
        c_out = ops.ones((2, 1, 1, 1), ms.float32)
        c_skip = ops.ones((2, 1, 1, 1), ms.float32)
        x = ops.ones((batch_size, 4, args.inputs.H // 8, args.inputs.W // 8), ms.float32)
        sigma_hat = ops.ones((), ms.float32)
        next_sigma = ops.ones((), ms.float32)
        s_in = ops.ones((1,), ms.float32)

        # create model
        text_encoder = model.conditioner
        unet = model.model
        unet = unet.to_float(ms.float32)
        vae = model.first_stage_model
        model_denoiser = model.denoiser

        scheduler_prepare_sampling_loop, scheduler_preprocess, denoiser, predict_noise, noisy_sample, vae_decoder = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if args.task == "text2img":
            data_prepare = Text2ImgEmbedder(text_encoder, vae, scheduler, model.scale_factor)
            model_export(
                net=data_prepare,
                inputs=(clip_tokens, time_tokens, clip_tokens, time_tokens, noise),
                name=args.inputs.data_prepare_model,
                model_save_path=model_save_path,
            )
        else:
            raise ValueError(f"Not support task: {args.task}")
        if scheduler_prepare_sampling_loop is None:
            scheduler_prepare_sampling_loop = SchedulerPrepareSamplingLoop(scheduler)
            model_export(
                net=scheduler_prepare_sampling_loop,
                inputs=noise,
                name=f"{args.inputs.prepare_sampling_loop}-{scheduler_type}",
                model_save_path=model_save_path,
            )
        if scheduler_preprocess is None:
            scheduler_preprocess = SchedulerPreModelInput(scheduler)
            model_export(
                net=scheduler_preprocess,
                inputs=(noise, ts, s_in),
                name=f"{args.inputs.scheduler_preprocess}-{scheduler_type}",
                model_save_path=model_save_path,
            )
        if denoiser is None:
            denoiser = Denoiser(model_denoiser)
            model_export(
                net=denoiser,
                inputs=(sigma_hat_s, 4),
                name=args.inputs.denoiser,
                model_save_path=model_save_path,
            )
        if predict_noise is None:
            predict_noise = PredictNoise(unet)
            model_export(
                net=predict_noise,
                inputs=(noised_input, c_noise, context, y),
                name=args.inputs.predict_noise_model,
                model_save_path=model_save_path,
            )
        if noisy_sample is None:
            noisy_sample = NoisySample(scheduler)
            model_export(
                net=noisy_sample,
                inputs=(model_output, c_out, noised_input, c_skip, scale, x, sigma_hat, next_sigma),
                name=f"{args.inputs.noisy_sample_model}-{scheduler_type}",
                model_save_path=model_save_path,
            )
        if vae_decoder is None:
            vae_decoder = VAEDecoder(vae, model.scale_factor)
            model_export(
                net=vae_decoder, inputs=noise, name=args.inputs.vae_decoder_model, model_save_path=model_save_path
            )


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
    parser.add_argument("--model", type=str, required=True, help="path to config which constructs model.")
    parser.add_argument("--only_converte_lite", default=False, type=str2bool, help="whether convert MindSpore mindir")
    parser.add_argument("--converte_lite", default=False, type=str2bool, help="whether convert lite mindir")
    parser.add_argument("--output_path", type=str, default="output", help="dir to write results to")
    parser.add_argument("--sampler", type=str, default="config/schedule/euler_edm.yaml", help="infer sampler yaml path")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--log_level", type=str, default="INFO", help="log level, options: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--ms_amp_level", type=str, default="O2")
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
        default_ckpt = "./models/sd_xl_base_1.0_ms.ckpt"
    else:
        raise ValueError(f"{args.task} is invalid, should be in [text2img, img2img, inpaint]")
    inputs = OmegaConf.load(inputs_config_path)

    key_settings_info = ["Key Settings:\n" + "=" * 50]
    key_settings_info += [
        f"SD export task: {args.task}",
        f"model config: {args.model}",
        f"inputs config: {inputs_config_path}",
        f"Number of samples in each trial: {args.n_samples}",
        f"Sampler: {args.sampler}",
    ]
    for key in inputs.keys():
        key_settings_info.append(f"{key}: {inputs[key]}")

    logger.info("\n".join(key_settings_info))

    args.inputs = inputs
    main(args)
