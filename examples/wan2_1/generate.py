# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import random
import sys
from datetime import datetime

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

import wan
from PIL import Image
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_image, cache_video, str2bool

import mindspore as ms
import mindspore.mint.distributed as dist
from mindspore.communication import GlobalComm

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
        "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
        "Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. "
        "The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. "
        "A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    # Size check
    assert (
        args.size in SUPPORTED_SIZES[args.task]
    ), f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a image or video from a text prompt or image using Wan")
    parser.add_argument(
        "--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run."
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1",
    )
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
    )
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--ring_size", type=int, default=1, help="The size of the ring attention parallelism in DiT.")
    parser.add_argument("--t5_zero3", action="store_true", default=False, help="Whether to use ZeRO3 for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_zero3", action="store_true", default=False, help="Whether to use ZeRO3 for DiT.")
    parser.add_argument("--save_file", type=str, default=None, help="The file to save the generated image or video to.")
    parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate the image or video from.")
    parser.add_argument("--use_prompt_extend", action="store_true", default=False, help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.",
    )
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="ch",
        choices=["ch", "en"],
        help="The target language of prompt extend.",
    )
    parser.add_argument("--base_seed", type=int, default=0, help="The seed to use for generating the image or video.")
    parser.add_argument("--image", type=str, default=None, help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="The solver used to sample."
    )
    parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers."
    )
    parser.add_argument("--sample_guide_scale", type=float, default=5.0, help="Classifier free guidance scale.")

    # extra for mindspore
    parser.add_argument("--ulysses_sp", action="store_true", default=False, help="turn on ulysses parallelism in DiT.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    if args.ulysses_sp or args.t5_zero3 or args.dit_zero3:
        dist.init_process_group(backend="hccl")
        ms.set_auto_parallel_context(parallel_mode="data_parallel")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if args.ulysses_sp:
            args.ulysses_size = world_size
    else:
        assert not (
            args.t5_zero3 or args.dit_zero3
        ), "t5_zero3 and dit_zero3 are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), "context parallel are not supported in non-distributed environments."
        rank = 0
        world_size = 1

    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert (
            args.ulysses_size * args.ring_size == world_size
        ), "The number of ulysses_size and ring_size should be equal to the world size."

    if args.use_prompt_extend:
        raise NotImplementedError("prompt_extend is not supported")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, "`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    # TODO: GlobalComm.INITED -> mint.is_initialzed
    if GlobalComm.INITED:
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            raise NotImplementedError

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            rank=rank,
            t5_zero3=args.t5_zero3,
            dit_zero3=args.dit_zero3,
            use_usp=args.ulysses_sp,
            t5_cpu=args.t5_cpu,
        )

        logging.info(f"Generating {'image' if 't2i' in args.task else 'video'} ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )

    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            raise NotImplementedError

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            rank=rank,
            t5_zero3=args.t5_zero3,
            dit_zero3=args.dit_zero3,
            use_usp=args.ulysses_sp,
            t5_cpu=args.t5_cpu,
        )

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = ".png" if "t2i" in args.task else ".mp4"
            args.save_file = (
                f"{args.task}_{args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}"
                + suffix
            )

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None], save_file=args.save_file, nrow=1, normalize=True, value_range=(-1, 1)
            )
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    # TODO: remove global seed
    ms.set_seed(args.base_seed)
    generate(args)
