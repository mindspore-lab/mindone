import argparse
import datetime
import logging
import os
import sys
import numpy as np
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import nn, ops
from transformers import CLIPTokenizer

# Set up the directory paths
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline
from pipeline.t2v_turbo_ms_pipeline import T2VTurboMSPipeline
from model_scope.unet_3d_condition import UNet3DConditionModel
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from tools.convert_weights import convert_lora, convert_t2v_vc2
from utils.common_utils import load_model_checkpoint
from utils.download import DownLoad
from utils.env import init_env
from utils.lora import collapse_lora, monkeypatch_remove_lora
from utils.lora_handler import LoraHandler
from utils.utils import instantiate_from_config
from utils.common_utils import set_torch_2_attn

from mindone.diffusers import AutoencoderKL
from mindone.transformers import CLIPTextModel
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger
from mindone.visualize.videos import save_videos

sys.path.append("../stable_diffusion_xl")
from gm.modules.embedders.open_clip.tokenizer import tokenize

# Logging configuration
logger = logging.getLogger(__name__)

# Model URLs and cache directories
MODEL_CACHE_VC2 = "model_cache/t2v-vc2/"
MODEL_CACHE_MS = "model_cache/t2v-ms/"
VC2_URL_VC2 = "https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt"
LORA_URL_VC2 = "https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2/blob/main/unet_lora.pt"
LORA_URL_MS = "https://huggingface.co/jiachenli-ucsb/T2V-Turbo-MS/blob/main/unet_lora.pt"
MODEL_URL_MS = "ali-vilab/text-to-video-ms-1.7b"

# Data type mapping
dtype_map = {"fp32": ms.float32, "fp16": ms.float16, "bf16": ms.bfloat16}


def download_model(args, model_type):
    """Download model based on the specified type."""
    return download_vc2(args) if model_type == "vc2" else download_ms(args)


def download_vc2(args):
    """Download VC2 model and LoRA weights."""
    unet_dir, t2v_dir = args.unet_dir, args.base_model_dir

    if not os.path.exists(unet_dir):
        logger.info(f"Downloading UNet model to {unet_dir}...")
        DownLoad().download_url(LORA_URL_VC2, path=MODEL_CACHE_VC2)
        convert_lora(
            src_path=os.path.join(MODEL_CACHE_VC2, "unet_lora.pt"),
            target_path=os.path.join(MODEL_CACHE_VC2, "unet_lora.ckpt"),
        )
        unet_dir = os.path.join(MODEL_CACHE_VC2, "unet_lora.ckpt")

    if not os.path.exists(t2v_dir):
        logger.info(f"Downloading base model to {t2v_dir}...")
        DownLoad().download_url(VC2_URL_VC2, path=MODEL_CACHE_VC2)
        convert_t2v_vc2(
            src_path=os.path.join(MODEL_CACHE_VC2, "model.ckpt"),
            target_path=os.path.join(MODEL_CACHE_VC2, "VideoCrafter2_model_ms.ckpt"),
        )
        t2v_dir = os.path.join(MODEL_CACHE_VC2, "VideoCrafter2_model_ms.ckpt")

    return t2v_dir, unet_dir


def download_ms(args):
    """Download MS model and LoRA weights."""
    unet_dir = args.unet_dir
    if not os.path.exists(unet_dir):
        logger.info(f"Downloading UNet model to {unet_dir}...")
        DownLoad().download_url(LORA_URL_MS, path=MODEL_CACHE_MS)
        convert_lora(
            src_path=os.path.join(MODEL_CACHE_MS, "unet_lora.pt"),
            target_path=os.path.join(MODEL_CACHE_MS, "unet_lora.ckpt"),
        )
        unet_dir = os.path.join(MODEL_CACHE_MS, "unet_lora.ckpt")

    return args.base_model_dir, unet_dir


def load_t2v_pipeline(args):
    """Load the Text-to-Video pipeline based on the selected model type."""
    t2v_dir, unet_dir = download_model(args, args.teacher)
    if args.teacher == "vc2":
        return load_vc2_pipeline(t2v_dir, unet_dir, args)
    else:
        return load_ms_pipeline(t2v_dir, unet_dir, args)


def load_vc2_pipeline(t2v_dir, unet_dir, args):
    """Load the VC2 pipeline."""
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())

    clip_dir = model_config["params"]["cond_stage_config"]["params"].get("pretrained_ckpt_path", None)
    if clip_dir:
        clip_dir = os.path.join(MODEL_CACHE_VC2, clip_dir)
        model_config["params"]["cond_stage_config"]["params"]["pretrained_ckpt_path"] = clip_dir

    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(pretrained_t2v, t2v_dir)

    unet_config = model_config["params"]["unet_config"]
    unet_config["params"].update({"time_cond_proj_dim": 256, "use_checkpoint": False})
    unet = instantiate_from_config(unet_config)
    ms.load_param_into_net(unet, pretrained_t2v.model.diffusion_model.parameters_dict(), False)

    # LoRA injection
    lora_manager = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=True,
        save_for_webui=True,
        unet_replace_modules=["UNetModel"],
    )
    lora_manager.add_lora_to_model(
        True,
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

    pretrained_t2v.set_train(False)

    # Prepare the pipeline
    scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )
    return T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)


def load_ms_pipeline(t2v_dir, unet_dir, args):
    """Load the MS pipeline."""
    dtype = dtype_map[args.dtype]
    if os.path.exists(t2v_dir):
        pretrained_model_path = t2v_dir
        logger.info(f"Using pretrained model from directory: {t2v_dir}")
    else:
        pretrained_model_path = MODEL_URL_MS
        logger.warning(f"Directory {t2v_dir} not found. Falling back to default model URL: {MODEL_URL_MS}")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    teacher_unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", dtype=dtype)

    unet = UNet3DConditionModel.from_config(teacher_unet.config, time_cond_proj_dim=256, dtype=dtype)
    ms.load_param_into_net(unet, teacher_unet.parameters_dict(), False)
    set_torch_2_attn(unet)

    # LoRA injection
    lora_manager = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=True,
        save_for_webui=True,
    )
    lora_manager.add_lora_to_model(
        True,
        unet,
        lora_manager.unet_replace_modules,
        lora_path=unet_dir,
        dropout=0.1,
        r=32,
    )
    collapse_lora(unet, lora_manager.unet_replace_modules)
    monkeypatch_remove_lora(unet)
    unet.set_train(False)

    # Prepare the pipeline
    noise_scheduler = T2VTurboScheduler()
    return T2VTurboMSPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
        dtype=dtype,
    )


def main(args):
    """Main entry point for the script."""
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") if args.append_timestr else ""
    save_dir = os.path.join(args.output_path, time_str) if time_str else args.output_path
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    latent_dir = os.path.join(args.output_path, "denoised_latents")
    if args.save_latent:
        os.makedirs(latent_dir, exist_ok=True)

    # Initialize environment
    rank_id, device_num = init_env(
        args.mode,
        args.seed,
        distributed=False,
        device_target=args.device_target,
        jit_level=args.jit_level,
        global_bf16=args.global_bf16,
        debug=args.debug,
    )

    # Prepare the Text-to-Video pipeline
    pipeline = load_t2v_pipeline(args)

    # Mixed-precision setup
    amp_level = "O2"
    if not args.global_bf16:
        for module in [pipeline.unet, pipeline.vae]:
            module = auto_mixed_precision(
                module,
                amp_level=amp_level,
                dtype=dtype_map[args.dtype],
                custom_fp32_cells=[nn.GroupNorm] if args.keep_gn_fp32 else [],
            )
        logger.info(f"Set mixed precision to O2 with dtype={args.dtype}")

    # Inference
    generator = np.random.Generator(np.random.PCG64(args.seed))
    prompt = args.prompt if args.teacher == "ms" else ms.Tensor(np.array(tokenize(args.prompt)[0], dtype=np.int32))

    result = pipeline(
        prompt=prompt,
        frames=args.num_frames,
        fps=args.fps,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_videos_per_prompt=1,
        generator=generator,
    )

    # Post-processing
    video = result[0]  # result -> (1, 3, 16, 320, 512)
    video = ops.clamp(video.float(), -1.0, 1.0)
    video = video.permute(1, 0, 2, 3)
    video = (video + 1.0) / 2.0
    video = video.permute(0, 2, 3, 1).asnumpy()

    # Save result
    out_path = os.path.join(save_dir, "out.mp4")
    save_videos(video, out_path, fps=args.fps / args.frame_interval)
    logger.info(f"Video saved in {out_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Text-to-Video Generation Pipeline")

    # Model and Configuration
    parser.add_argument(
        "--config", "-c", default="configs/inference_t2v_512_v2.0.yaml", type=str, help="Path to config YAML file."
    )
    parser.add_argument(
        "--teacher", type=str, choices=["vc2", "ms"], required=True, help="Select the model type: 'vc2' or 'ms'."
    )
    parser.add_argument(
        "--unet_dir", type=str, default="model_cache/t2v-vc2/unet_lora.ckpt", help="Directory of the UNet model."
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="model_cache/t2v-vc2/VideoCrafter2_model_ms.ckpt",
        help="Directory of the VideoCrafter2 checkpoint.",
    )

    # Data Type and Precision
    parser.add_argument("--dtype", default="fp16", type=str, choices=["bf16", "fp16"], help="Data type to use.")
    parser.add_argument("--global_bf16", default=False, type=str2bool, help="Use bf16 if supported.")
    parser.add_argument("--keep_gn_fp32", default=True, type=str2bool, help="Keep GroupNorm in fp32.")
    parser.add_argument(
        "--jit_level", default="O0", type=str, choices=["O0", "O1", "O2"], help="Compilation optimization level."
    )

    # Video Generation Parameters
    parser.add_argument("--image_size", type=int, default=256, nargs="+", help="Image size.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames.")
    parser.add_argument("--fps", type=int, default=8, help="FPS of the output video.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Scale for classifier-free guidance.")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of denoising steps.")
    parser.add_argument("--frame_interval", default=1, type=int, help="Frame sampling frequency.")

    # Input Prompt
    parser.add_argument("--prompt", type=str, default="A dancing cat.", help="Input prompt for generation.")

    # Output Settings
    parser.add_argument("--output_path", type=str, default="samples", help="Output directory for generated videos.")
    parser.add_argument("--save_latent", type=str2bool, default=False, help="Save denoised video latent.")
    parser.add_argument("--append_timestr", type=str2bool, default=True, help="Append timestamp to output path.")

    # Miscellaneous
    parser.add_argument("--device_target", type=str, default="Ascend", help="Device target (Ascend or GPU).")
    parser.add_argument("--mode", type=int, default=0, help="Execution mode (GRAPH_MODE or PYNATIVE_MODE).")
    parser.add_argument("--debug", type=str2bool, default=False, help="Debug mode.")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
