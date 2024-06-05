import argparse
import logging
import os
import sys
from typing import Tuple

import yaml

import mindspore as ms
from mindspore import nn
from mindspore.communication.management import get_group_size, get_rank, init

# TODO: remove in future when mindone is ready for install
mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))
from opensora.models.diffusion.latte.modeling_latte import LatteT2V, LayerNorm
from opensora.models.diffusion.latte.modules import Attention
from opensora.utils.utils import _check_cfgs_in_parser, get_precision

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    parallel_mode: str = "data",
    enable_dvm: bool = False,
    precision_mode: str = None,
    global_bf16: bool = False,
) -> Tuple[int, int, int]:
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )
        if parallel_mode == "optim":
            print("use optim parallel")
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                enable_parallel_optimizer=True,
            )
            init()
            device_num = get_group_size()
            rank_id = get_rank()
        else:
            init()
            device_num = get_group_size()
            rank_id = get_rank()
            logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
            ms.reset_auto_parallel_context()

            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
                device_num=device_num,
            )

        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        logger.info(dict(zip(var_info, var_value)))

    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )

    if enable_dvm:
        print("enable dvm")
        ms.set_context(enable_graph_kernel=True, graph_kernel_flags="--disable_cluster_ops=Pow,Select")
    if precision_mode is not None and len(precision_mode) > 0:
        ms.set_context(ascend_config={"precision_mode": precision_mode})
    if global_bf16:
        print("Using global bf16")
        ms.set_context(
            ascend_config={"precision_mode": "allow_mix_precision_bf16"}
        )  # reset ascend precison mode globally
    return rank_id, device_num


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument("--model_path", type=str, default="LanguageBind/Open-Sora-Plan-v1.0.0")
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=None,
        help="If not provided, will search for ckpt file under `model_path`"
        "If provided, will use this pretrained ckpt path.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="65x512x512",
        help="Model version in ['65x512x512', '221x512x512', '513x512x512'] ",
    )
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--ae", type=str, default="CausalVAEModel_4x8x8")

    parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")

    parser.add_argument("--guidance_scale", type=float, default=7.5, help="the scale for classifier-free guidance")

    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50, help="Diffusion Sampling Steps")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--text_prompt",
        type=str,
        nargs="+",
        help="A list of text prompts to be generated with. Also allow input a txt file or csv file.",
    )
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument(
        "--force_images", default=False, type=str2bool, help="Whether to generate images given text prompts"
    )
    parser.add_argument("--enable_tiling", action="store_true", help="whether to use vae tiling to save memory")
    parser.add_argument(
        "--enable_time_chunk",
        type=str2bool,
        default=False,
        help="Whether to enable vae decoding to process temporal frames as small, overlapped chunks. Defaults to False.",
    )

    parser.add_argument("--batch_size", default=1, type=int, help="batch size for dataloader")
    # MS new args
    parser.add_argument("--device", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--parallel_mode", default="data", type=str, choices=["data", "optim"], help="parallel mode: data, optim"
    )
    parser.add_argument("--enable_dvm", default=False, type=str2bool, help="enable dvm mode")

    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--global_bf16", action="store_true", help="whether to enable gloabal bf16 for diffusion model training."
    )
    parser.add_argument(
        "--vae_precision",
        default="bf16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for vae. Default is `bf16`, which corresponds to ms.bfloat16",
    )
    parser.add_argument(
        "--text_encoder_precision",
        default="bf16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for T5 text encoder. Default is `bf16`, which corresponds to ms.bfloat16",
    )
    parser.add_argument(
        "--amp_level", type=str, default="O2", help="Set the amp level for the transformer model. Defaults to O2."
    )
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument(
        "--num_videos_per_prompt", type=int, default=1, help="the number of images to be generated for each prompt"
    )
    parser.add_argument(
        "--save_latents",
        action="store_true",
        help="Whether to save latents (before vae decoding) instead of video files.",
    )
    parser.add_argument(
        "--decode_latents",
        action="store_true",
        help="whether to load the existing latents saved in npy files and run vae decoding",
    )
    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 1. init env
    args = parse_args()
    save_dir = args.save_img_path
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)
    # 1. init
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        enable_dvm=args.enable_dvm,
        precision_mode=args.precision_mode,
        global_bf16=args.global_bf16,
    )

    # 4. latte model initiate and weight loading
    logger.info(f"Latte-{args.version} init")
    transformer_model = LatteT2V.from_pretrained(
        args.model_path,
        subfolder=args.version,
        checkpoint_path=args.pretrained_ckpt,
        enable_flash_attention=args.enable_flash_attention,
    )
    transformer_model.force_images = args.force_images
    # mixed precision
    dtype = get_precision(args.precision)
    if args.precision in ["fp16", "bf16"]:
        if not args.global_bf16:
            amp_level = args.amp_level
            transformer_model = auto_mixed_precision(
                transformer_model,
                amp_level=args.amp_level,
                dtype=dtype,
                custom_fp32_cells=[LayerNorm, Attention, nn.SiLU, nn.GELU] if dtype == ms.float16 else [],
            )
            logger.info(f"Set mixed precision to O2 with dtype={args.precision}")
        else:
            logger.info(f"Using global bf16 for latte t2v model. Force model dtype from {dtype} to ms.bfloat16")
            dtype = ms.bfloat16
    elif args.precision == "fp32":
        amp_level = "O0"
    else:
        raise ValueError(f"Unsupported precision {args.precision}")

    transformer_model = transformer_model.set_train(False)
    for param in transformer_model.get_parameters():  # freeze transformer_model
        param.requires_grad = False

    from mindspore import ops

    bs = 1
    latent_size = 64
    num_frames = 17
    latent_model_input = ops.randn((bs, num_frames, 4, latent_size, latent_size))  # b t c h w
    prompt_embeds = ops.randn((bs, 300, 4096))
    prompt_embeds_mask = ops.zeros((bs, 300))
    prompt_embeds_mask[:, -10:] = 1
    current_timestep = ms.Tensor([1], dtype=ms.int32)
    noise_pred = transformer_model(
        latent_model_input,
        encoder_hidden_states=prompt_embeds,
        timestep=current_timestep,
        added_cond_kwargs={},
        enable_temporal_attentions=True,
        encoder_attention_mask=prompt_embeds_mask,
    )
