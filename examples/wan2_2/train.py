# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import random
import sys

from PIL import Image

import mindspore as ms
import mindspore.mint.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.trainer import LoRATrainer, create_video_dataset
from wan.utils.utils import str2bool

EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
        "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
        "Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. "
        "The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. "
        "A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupported task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    # Size check
    assert (
        args.size in SUPPORTED_SIZES[args.task]
    ), f"Unsupported size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a image or video from a text prompt or image using Wan")
    parser.add_argument(
        "--task", type=str, default="t2v-A14B", choices=list(WAN_CONFIGS.keys()), help="The task to run."
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
    )
    parser.add_argument(
        "--frame_num", type=int, default=None, help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
    )
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--t5_zero3", action="store_true", default=False, help="Whether to use ZeRO3 for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_zero3", action="store_true", default=False, help="Whether to use ZeRO3 for DiT.")
    parser.add_argument("--save_file", type=str, default=None, help="The file to save the generated video to.")
    parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate the video from.")
    parser.add_argument("--base_seed", type=int, default=-1, help="The seed to use for generating the video.")
    parser.add_argument("--image", type=str, default=None, help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="The solver used to sample."
    )
    parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers."
    )
    parser.add_argument("--sample_guide_scale", type=float, default=None, help="Classifier free guidance scale.")
    parser.add_argument("--text_dropout_rate", type=float, default=0.1, help="The dropout rate for text encoder.")
    parser.add_argument("--validation_interval", type=int, default=100, help="The interval for validation.")
    parser.add_argument("--save_interval", type=int, default=100, help="The interval for saving checkpoints.")
    parser.add_argument("--output_dir", type=str, default="./output", help="The output directory to save checkpoints.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The learning rate for training.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="The weight decay for training.")
    parser.add_argument("--data_root", type=str, default=None, help="The root directory of training data.")
    parser.add_argument("--dataset_file", type=str, default=None, help="The dataset file for training data.")
    parser.add_argument("--caption_column", type=str, default="caption", help="The caption column in the dataset file.")
    parser.add_argument("--video_column", type=str, default="video", help="The video column in the dataset file.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for training.")
    parser.add_argument(
        "--text_drop_prob", type=float, default=0.1, help="The probability of dropping text input during training."
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="The number of epochs for training.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="The maximum gradient norm for clipping.")

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


def train(args):
    rank = int(os.getenv("RANK_ID", 0))
    world_size = int(os.getenv("RANK_SIZE", 1))
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if args.offload_model:
        raise ValueError("offload_model is not supported in training currently.")

    if world_size > 1:
        dist.init_process_group(backend="hccl", init_method="env://", rank=rank, world_size=world_size)
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    else:
        assert not (
            args.t5_zero3 or args.dit_zero3
        ), "t5_zero3 and dit_zero3 are not supported in non-distributed environments."
        assert not (args.ulysses_size > 1), "sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, "The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert (
            cfg.num_heads % args.ulysses_size == 0
        ), f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Training job args: {args}")
    logging.info(f"Training model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    if "t2v" in args.task:
        raise NotImplementedError
    elif "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            rank=rank,
            t5_zero3=args.t5_zero3,
            dit_zero3=args.dit_zero3,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=True,
        )

        logging.info("Prepare trainer ...")
        size_buckets = tuple(SIZE_CONFIGS[size] for size in SUPPORTED_SIZES[args.task])
        train_loader = create_video_dataset(
            data_root=args.data_root,
            dataset_file=args.dataset_file,
            caption_column=args.caption_column,
            video_column=args.video_column,
            frame_num=args.frame_num,
            size_buckets=size_buckets,
            batch_size=args.batch_size,
            num_shards=world_size,
            shard_id=rank,
            text_drop_prob=args.text_drop_prob,
        )
        training_config = dict(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_interval=args.validation_interval,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
            frame_num=args.frame_num,
            num_train_timesteps=cfg.num_train_timesteps,
            vae_stride=cfg.vae_stride,
            patch_size=cfg.patch_size,
            max_grad_norm=args.max_grad_norm,
        )
        generation_config = dict(
            input_prompt=args.prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )
        trainer = LoRATrainer(wan_ti2v, train_loader, training_config, generation_config)

        logging.info("Start training ...")
        trainer.train(args.num_epochs)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = _parse_args()
    train(args)
