import argparse
import logging
import os
import sys

from omegaconf import OmegaConf

import mindspore as ms
from mindspore import ops

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(workspace))
from ldm.modules.logger import set_logger
from ldm.util import instantiate_from_config, str2bool
from libs.helper import load_model_from_config, set_env
from libs.infer_engine.export_modules import (
    Img2ImgDataPrepare,
    InpaintDataPrepare,
    InpaintPredictNoise,
    NoisySample,
    PredictNoise,
    Text2ImgDataPrepare,
    VAEDecoder,
)

logger = logging.getLogger("text to image speed up")


def model_export(net, inputs, name, model_save_path):
    ms.export(net, *inputs, file_name=os.path.join(model_save_path, name), file_format="MINDIR")
    logger.info(f"convert {name} mindir done")


def lite_convert(name, model_save_path, converter):
    import mindspore_lite as mslite

    mindir_path = os.path.join(model_save_path, f"{name}.mindir")
    if not os.path.exists(mindir_path):
        mindir_path = os.path.join(model_save_path, f"{name}_graph.mindir")
    converter.convert(
        fmk_type=mslite.FmkType.MINDIR,
        model_file=mindir_path,
        output_file=os.path.join(model_save_path, f"{name}_lite"),
        config_file="./libs/infer_engine/sd_lite.cfg",
    )
    logger.info(f"convert {name} lite mindir done")


def main(args):
    # set logger
    set_env(args)
    ms.set_context(device_target=args.device_target)
    # create model
    config = OmegaConf.load(f"{args.model}")
    model = load_model_from_config(
        config,
        ckpt=args.inputs.ckpt_path,
        use_lora=args.inputs.use_lora,
        lora_rank=args.inputs.lora_rank,
        lora_only_ckpt=args.inputs.lora_ckpt_path,
    )
    args.model_save_path = f"{config.model.name}-{args.task}"
    sampler_config = OmegaConf.load(args.sampler)
    scheduler = instantiate_from_config(sampler_config)
    scheduler_type = sampler_config.type

    # data
    batch_size = args.n_samples
    tokenized_prompts = ops.ones((batch_size, 77), ms.int32)
    output_dim = 768 if args.version.startswith("1.") else 1024
    prompt_embeds = ops.ones((batch_size, 77, output_dim), ms.float16)
    noise = ops.ones((batch_size, 4, args.inputs.H // 8, args.inputs.W // 8), ms.float16)
    ts = ops.ones((), ms.int32)
    img = ops.ones((batch_size, 3, args.inputs.H, args.inputs.W), ms.float16)
    mask = ops.ones((batch_size, 1, args.inputs.H, args.inputs.W), ms.float16)

    # create model
    text_encoder = model.cond_stage_model
    unet = model.model
    vae = model.first_stage_model

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
        if args.task == "text2img":
            data_prepare = Text2ImgDataPrepare(text_encoder, vae, scheduler, model.scale_factor)
            model_export(
                net=data_prepare,
                inputs=(tokenized_prompts, tokenized_prompts, noise),
                name=args.inputs.data_prepare_model,
                model_save_path=model_save_path,
            )

            predict_noise = PredictNoise(unet, args.scale)
            model_export(
                net=predict_noise,
                inputs=(noise, ts, prompt_embeds, prompt_embeds),
                name=args.inputs.predict_noise_model,
                model_save_path=model_save_path,
            )

            noisy_sample = NoisySample(scheduler)
            model_export(
                net=noisy_sample,
                inputs=(noise, ts, noise, ts),
                name=f"{args.inputs.noisy_sample_model}-{scheduler_type}",
                model_save_path=model_save_path,
            )

            vae_decoder = VAEDecoder(vae, model.scale_factor)
            model_export(
                net=vae_decoder, inputs=(noise,), name=args.inputs.vae_decoder_model, model_save_path=model_save_path
            )

        elif args.task == "img2img":
            data_prepare = Img2ImgDataPrepare(text_encoder, vae, scheduler, model.scale_factor)
            model_export(
                net=data_prepare,
                inputs=(tokenized_prompts, tokenized_prompts, img, noise, ts),
                name=args.inputs.data_prepare_model,
                model_save_path=model_save_path,
            )

            predict_noise = PredictNoise(unet, args.scale)
            model_export(
                net=predict_noise,
                inputs=(noise, ts, prompt_embeds, prompt_embeds),
                name=args.inputs.predict_noise_model,
                model_save_path=model_save_path,
            )

            noisy_sample = NoisySample(scheduler)
            model_export(
                net=noisy_sample,
                inputs=(noise, ts, noise, ts),
                name=f"{args.inputs.noisy_sample_model}-{scheduler_type}",
                model_save_path=model_save_path,
            )

            vae_decoder = VAEDecoder(vae, model.scale_factor)
            model_export(
                net=vae_decoder, inputs=(noise,), name=args.inputs.vae_decoder_model, model_save_path=model_save_path
            )
        elif args.task == "inpaint":
            data_prepare = InpaintDataPrepare(text_encoder, vae, scheduler, model.scale_factor)
            model_export(
                net=data_prepare,
                inputs=(tokenized_prompts, tokenized_prompts, img, mask, noise),
                name=args.inputs.data_prepare_model,
                model_save_path=model_save_path,
            )
            c_concat = ops.ones((batch_size, 5, args.inputs.H // 8, args.inputs.W // 8), ms.float16)
            c_crossattn = ops.ones((batch_size * 2, 77, output_dim), ms.float16)
            predict_noise = InpaintPredictNoise(unet, args.scale)
            model_export(
                net=predict_noise,
                inputs=(noise, ts, c_concat, c_crossattn),
                name=args.inputs.predict_noise_model,
                model_save_path=model_save_path,
            )

            noisy_sample = NoisySample(scheduler)
            model_export(
                net=noisy_sample,
                inputs=(noise, ts, noise, ts),
                name=f"{args.inputs.noisy_sample_model}-{scheduler_type}",
                model_save_path=model_save_path,
            )

            vae_decoder = VAEDecoder(vae, model.scale_factor)
            model_export(
                net=vae_decoder, inputs=(noise,), name=args.inputs.vae_decoder_model, model_save_path=model_save_path
            )
        else:
            raise ValueError(f"Not support task: {args.task}")

    if args.converte_lite:
        lite_convert(args.inputs.data_prepare_model, model_save_path, converter)
        lite_convert(args.inputs.predict_noise_model, model_save_path, converter)
        lite_convert(f"{args.inputs.noisy_sample_model}-{scheduler_type}", model_save_path, converter)
        lite_convert(args.inputs.vae_decoder_model, model_save_path, converter)


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
        help="Task name, should be [text2img, img2img], "
        "if choose a task name, use the config/[task].yaml for inputs",
        choices=["text2img", "img2img"],
    )
    parser.add_argument("--model", type=str, required=True, help="path to config which constructs model.")
    parser.add_argument("--only_converte_lite", default=False, type=str2bool, help="whether convert MindSpore mindir")
    parser.add_argument("--converte_lite", default=True, type=str2bool, help="whether convert lite mindir")
    parser.add_argument("--output_path", type=str, default="output", help="dir to write results to")
    parser.add_argument("--sampler", type=str, default="config/schedule/ddim.yaml", help="infer sampler yaml path")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="unconditional guidance scale: "
        "eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
        "Simplified: `uc + scale * (uc - prompt)`",
    )
    parser.add_argument("-v", "--version", type=str, default="2.0", help="Stable diffusion version, 1.x or 2.0.")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--log_level", type=str, default="INFO", help="log level, options: DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()
    set_logger(name="", output_dir=args.output_path, rank=0, log_level=args.log_level)

    if args.version:
        os.environ["SD_VERSION"] = args.version
    if args.scale is None:
        args.scale = 7.5 if args.version.startswith("1.") else 9.0
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
    else:
        raise ValueError(f"{args.task} is invalid, should be in [text2img, img2img, inpaint]")
    inputs = OmegaConf.load(inputs_config_path)
    if not inputs.ckpt_path:
        inputs.ckpt_path = default_ckpt
        logger.info(f"ckpt_path in {inputs_config_path} not set, set default ckpt {default_ckpt}")
    if not os.path.exists(inputs.ckpt_path):
        raise ValueError(f"{inputs.ckpt_path} is not exist!, please set it in {inputs_config_path}")
    if inputs.use_lora and not os.path.exists(inputs.lora_ckpt_path):
        raise ValueError(f"{inputs.lora_ckpt_path} is not exist when use lora, please set it in {inputs_config_path}")

    key_settings_info = ["Key Settings:\n" + "=" * 50]
    key_settings_info += [
        f"SD export task: {args.task}",
        f"model config: {args.model}",
        f"inputs config: {inputs_config_path}",
        f"Number of samples in each trial: {args.n_samples}",
        f"Sampler: {args.sampler}",
        f"Uncondition guidance scale: {args.scale}",
    ]
    for key in inputs.keys():
        key_settings_info.append(f"{key}: {inputs[key]}")

    logger.info("\n".join(key_settings_info))

    args.inputs = inputs
    main(args)
