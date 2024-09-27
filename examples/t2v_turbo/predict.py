# Prediction interface for Cog ⚙️
# https://cog.run/python

import os, sys
import argparse
import datetime
import subprocess
import time
import logging
import numpy as np

from omegaconf import OmegaConf

import mindspore as ms
from mindspore import nn, mint

from mindone.utils.logger import set_logger
from mindone.visualize.videos import save_videos
from mindone.utils.config import str2bool
from mindone.utils.amp import auto_mixed_precision

from utils.lora import collapse_lora, monkeypatch_remove_lora
from utils.common_utils import load_model_checkpoint
from utils.utils import instantiate_from_config
from utils.env import init_env
from utils.lora_handler import LoraHandler
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline

sys.path.append("./mindone/examples/stable_diffusion_xl")
from gm.modules.embedders.open_clip.tokenizer import tokenize

logger = logging.getLogger(__name__)
MODEL_URL = "https://weights.replicate.delivery/default/Ji4chenLi/t2v-turbo.tar"
MODEL_CACHE = "checkpoints/t2v-vc2/"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def main(args):
    if args.append_timestr:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir = f"{args.output_path}/{time_str}"
    else:
        save_dir = f"{args.output_path}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    dtype_map = {"fp32": ms.float32, "fp16": ms.float16, "bf16": ms.bfloat16}
    dtype = dtype_map[args.dtype]

    latent_dir = os.path.join(args.output_path, "denoised_latents")
    if args.save_latent:
        os.makedirs(latent_dir, exist_ok=True)

    # 1. init env
    rank_id, device_num = init_env(
        args.mode,
        args.seed,
        args.use_parallel,
        device_target=args.device_target,
        jit_level=args.jit_level,
        global_bf16=args.global_bf16,
        debug=args.debug,
    )

    # 2. model initiate and weight loading

    # if not os.path.exists(MODEL_CACHE):
    #     download_weights(MODEL_URL, MODEL_CACHE)

    base_model_dir = os.path.join(MODEL_CACHE, "t2v_VC2.ckpt")
    unet_dir = os.path.join(MODEL_CACHE, "unet_lora_vc2.pt")

    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(pretrained_t2v, base_model_dir)

    unet_config = model_config["params"]["unet_config"]
    unet_config["params"]["time_cond_proj_dim"] = 256
    unet = instantiate_from_config(unet_config)
    ms.load_param_into_net(unet, pretrained_t2v.model.diffusion_model.parameters_dict(), False)

    use_unet_lora = True
    lora_manager = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=use_unet_lora,
        save_for_webui=True,
        unet_replace_modules=["UNetModel"],
    )
    lora_manager.add_lora_to_model(
        use_unet_lora,
        unet,
        lora_manager.unet_replace_modules,
        lora_path=unet_dir,
        dropout=0.1,
        r=64,
    )
    unet.set_train(False)
    collapse_lora(unet, lora_manager.unet_replace_modules)
    monkeypatch_remove_lora(unet)

    if args.dtype != "fp32":
        unet_config["params"]["dtype"] = args.dtype
        new_unet = instantiate_from_config(unet_config)
        ms.load_param_into_net(new_unet, unet.parameters_dict(), False)
        pretrained_t2v.model.diffusion_model = new_unet
        del unet, new_unet

    else:
        pretrained_t2v.model.diffusion_model = unet
        del unet

    pretrained_t2v.set_train(False)

    # 2.1 amp
    if args.dtype not in ["fp32", "bf16"]:
        amp_level = "O2"
        if not args.global_bf16:
            pretrained_t2v = auto_mixed_precision(
                pretrained_t2v,
                amp_level=amp_level,
                dtype=dtype_map[args.dtype],
                custom_fp32_cells=[nn.GroupNorm] if args.keep_gn_fp32 else [],
            )
        logger.info(f"Set mixed precision to O2 with dtype={args.dtype}")
    else:
        amp_level = "O0"

    # 2.2 pipeline
    scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )
    pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)

    # 3. inference
    generator = np.random.Generator(np.random.PCG64(args.seed))

    # 3.1 tokenize
    tokens, _ = tokenize(args.prompt)
    tokens = ms.Tensor(np.array(tokens, dtype=np.int32))

    result = pipeline(
        prompt=tokens,
        frames=args.num_frames,
        fps=args.fps,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_videos_per_prompt=1,
        generator=generator,
    )

    # 4. post-processing

    video = result[0]  # result -> (1, 3, 16, 320, 512)
    video = mint.clamp(video.float(), -1.0, 1.0)
    video = video.permute(1, 0, 2, 3)
    video = (video + 1.0) / 2.0
    video = video.permute(0, 2, 3, 1).asnumpy()

    # 5. save result
    out_path = "./results/out.mp4"
    save_videos(video, out_path, fps=args.fps / args.frame_interval)

    logger.info(f"Video saved in {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="configs/inference_t2v_512_v2.0.yaml",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp32`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--global_bf16",
        default=False,
        type=str2bool,
        help="Experimental. If True, dtype will be overrided, operators will be computered in bf16 if they are supported by CANN",
    )
    parser.add_argument(
        "--keep_gn_fp32",
        default=True,
        type=str2bool,
        help="whether to keep GrounpNorm in fp32",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        nargs="+",
        help="image size in [256, 512]",
    )
    parser.add_argument("--num_frames", type=int, default=16, help="number of frames")
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports [“O0”, “O1”, “O2”]."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="the scale for classifier-free guidance",
    )
    parser.add_argument(
        "--guidance_channels",
        type=int,
        help="How many channels to use for classifier-free diffusion. If None, use half of the latent channels",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=4, help="Number of denoising steps"
    )
    parser.add_argument(
        "--frame_interval",
        default=1,
        type=int,
        help="Frames sampling frequency. Final video FPS will be equal to FPS / frame_interval.",
    )
    parser.add_argument("--fps", type=int, default=8, help="FPS of the output video")
    parser.add_argument(
        "--output_path",
        type=str,
        default="samples",
        help="output dir to save the generated videos",
    )
    parser.add_argument(
        "--save_latent",
        type=str2bool,
        default=False,
        help="Save denoised video latent. If True, the denoised latents will be saved in $output_path/denoised_latents",
    )
    parser.add_argument(
        "--append_timestr",
        type=str2bool,
        default=True,
        help="If true, an subfolder named with timestamp under output_path will be created to save the sampling results",
    )

    # inputs
    parser.add_argument(
        "--prompt",
        type=str,
        default="A dancing cat.",
        help="Input prompt for generation.",
    )

    # MS new args
    parser.add_argument(
        "--device_target", type=str, default="Ascend", help="Ascend or GPU"
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)",
    )
    parser.add_argument(
        "--use_parallel", default=False, type=str2bool, help="use parallel"
    )
    parser.add_argument(
        "--debug", type=str2bool, default=False, help="Execute inference in debug mode."
    )
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
