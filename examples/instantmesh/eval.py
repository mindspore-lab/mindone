"""Eval script using the model stage 1 trained ckpt to conduct arbitral novel view synthesis.

Design methdology: Unlike the ms translation that has been done for training,
we make the eval here more similar to the inference script below with np utilization.
Because for training, the np data proc parts should be translated into ms as much as possible
(see ~/examples/opensora_hpcai/opensora/datasets/video_dataset_refactored.py),
but not for the inference.

Thus we can do np for all data proc here to avoid tedious ms tranlsation of data.
Refer to inference.py for the full stage inference.
"""
import argparse
import datetime
import os
import sys
import time

import mindspore as ms
from mindspore import mint

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../../")))  # for loading mindone
# from loguru import logger
import logging

from omegaconf import OmegaConf
from transformers import ViTImageProcessor
from utils.eval_util import init_inference_env, make_grid_ms, save_image_ms, str2bool

from mindone.utils.config import instantiate_from_config
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)

from typing import Optional


def evaluate(args, epoch_num: Optional[str]):
    if args.append_timestr:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir = f"{args.output_path}/{time_str}"
    else:
        save_dir = f"{args.output_path}"
    image_path = save_dir
    if not args.debug:
        os.makedirs(image_path, exist_ok=True)

    rid, dnum = init_inference_env(
        args.mode,
        args.seed,
        device_target=args.device_target,
        device_id=int(os.getenv("DEVICE_ID")),
        jit_level=args.jit_level,
        debug=args.debug,
    )
    set_random_seed(42)
    set_logger(name="", output_dir=args.output_path, rank=rid, log_level=eval(args.log_level))

    # valdata preparation
    config = OmegaConf.load(args.datacfg)
    data = instantiate_from_config(config.data.val)
    valset_len = data.__len__()

    # init model & load ckpt
    config = OmegaConf.load(args.modelcfg)
    config.model.params.lrm_generator_config.params.dtype = args.dtype
    config.model.params.lrm_ckpt_path = args.itmh_ckpt

    model = instantiate_from_config(config.model)

    # create img name
    if args.itmh_ckpt.split("/")[-1] == "instant_nerf_large_ms.ckpt":
        image_path = os.path.join(image_path, "val_official_instantmesh_ckpt.png")
    else:
        if not epoch_num:  # train_resume.ckpt
            epoch_num = ms.load_checkpoint(args.itmh_ckpt).get(
                "epoch_num", 0
            )  # 0 means that there is no this key in the resume ckpt file
            image_path = os.path.join(image_path, f"val_e{epoch_num}.png")

    validation_step_outputs = []
    batches_time = []
    for index in range(0, valset_len, args.batch_size):
        val_batch_np = data.__getitem__(index)

        # [torch] prepare_validation_batch_data():
        # prepare for validation batch/mv2mesh inference(): see raw repo TODO del this comment once this script finishes
        val_input = model.prepare_validation_batch_data(
            val_batch_np,
            render_size=config.eval_render_size,
            _use_dataloader=False,
        )
        images = val_input["images"]

        # [torch] forward():
        # RGB image with [0,1] scale and properly sized requested by the ViTImgProc
        if images.ndim == 5:
            (B, N, C, H, W) = images.shape
            images = images.reshape(B * N, C, H, W)

        # ViTImageProcessor moved out from dino wrapper to the main here, to avoid being in .construct(), do ImageNetStandard normalization
        img_processor = ViTImageProcessor.from_pretrained(
            config.model.params.lrm_generator_config.params.encoder_model_name
        )
        images = img_processor(
            images=images,
            return_tensors="np",
            do_rescale=False,
            do_resize=False,
        )["pixel_values"]
        val_input["images"] = ms.Tensor(images).reshape(B, N, C, H, W)

        # infer
        start_time = time.time()
        render_images, render_alphas = model.forward_nocalloss(**val_input)

        batch_time = time.time() - start_time
        batches_time.append(batch_time)
        logger.info(f"Batch time cost: {batch_time: .3f}s.")
        # save result both img and alpha, in validation_step()
        # render_images = rearrange(render_images, 'b n c h w -> b c h (n w)')
        render_images = mint.permute(render_images, dims=(0, 2, 3, 1, 4)).flatten(start_dim=-2)
        validation_step_outputs.append(render_images)

    mean_time = sum(batches_time) / len(batches_time)
    logger.info(f"Mean Batch time: {mean_time: .3f}s.")
    # save mviews outputs
    images = mint.cat(validation_step_outputs, dim=0)  # enable for multiple batches

    # images = rearrange(images, 'r b c h w -> (r b) c h w')
    # images = images.flatten(start_dim=0, end_dim=1)
    assert len(images.shape) == 4, "images' shape not matched"

    grid = make_grid_ms(images, nrow=1, normalize=True, value_range=(0, 1))
    if not args.debug:
        save_image_ms(grid, image_path)
        logger.info(f"Saved image to {image_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--itmh_ckpt", default="CKPT_PATH")
    parser.add_argument(
        "--debug",
        # default=True,  # also setting debug as true will set pynative sync as true as well
        default=False,  # also setting debug as true will set pynative sync as true as well
        help="When debugging, set it true, to avoid saving too many ckpts and burn out the storage.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output",
        help="output dir to save the generated videos",
    )
    parser.add_argument(
        "--append_timestr",
        type=str2bool,
        default=True,
        help="If true, an subfolder named with timestamp under output_path will be created to save the sampling results",
    )
    parser.add_argument("--datacfg", default="configs/instant-nerf-large-train.yaml")
    parser.add_argument("--modelcfg", default="configs/instant-nerf-large-train.yaml")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--seed", type=int, default=42, help="Inference seed")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
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
    parser.add_argument("--batch_size", default=1, type=int, help="infer batch size")
    parser.add_argument(
        "--dtype",
        default="fp32",  # if amp level O0/1, must pass fp32
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what computation data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.itmh_ckpt.split("/")[-1] == "train_resume.ckpt":
        epoch_num = None
    else:
        epoch_num = args.itmh_ckpt.split("-e")[-1].split(".")[0]
    evaluate(args, epoch_num)
