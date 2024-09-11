import argparse
import logging
import os
import sys

import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import nn

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger
from mindone.visualize.videos import save_videos

sys.path.append(".")
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from opensora.models.ae import getae_wrapper
from opensora.models.ae.videobase.dataset_videobase import VideoDataset, create_dataloader
from opensora.models.ae.videobase.modules.updownsample import TrilinearInterpolate
from opensora.utils.ms_utils import init_env
from opensora.utils.utils import get_precision

logger = logging.getLogger(__name__)


def transform_to_rgb(x, rescale_to_uint8=True):
    x = np.clip(x, -1, 1)
    x = (x + 1) / 2
    if rescale_to_uint8:
        x = (255 * x).astype(np.uint8)
    return x


def main(args):
    real_video_dir = args.real_video_dir
    generated_video_dir = args.generated_video_dir

    resolution = args.resolution
    crop_size = args.crop_size
    num_frames = args.num_frames
    sample_rate = args.sample_rate
    sample_fps = args.sample_fps
    batch_size = args.batch_size
    num_workers = args.num_workers
    assert args.dataset_name == "video", "Only support video reconstruction!"
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        precision_mode=args.precision_mode,
        sp_size=args.sp_size,
        jit_level=args.jit_level,
    )

    if not os.path.exists(args.generated_video_dir):
        os.makedirs(args.generated_video_dir, exist_ok=True)

    set_logger(name="", output_dir=args.generated_video_dir, rank=0)

    kwarg = {"model_config": args.model_config}
    vae = getae_wrapper(args.ae)(args.ckpt, **kwarg)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor

    vae.set_train(False)
    for param in vae.get_parameters():
        param.requires_grad = False
    if args.precision in ["fp16", "bf16"]:
        amp_level = "O2"
        dtype = get_precision(args.precision)
        custom_fp32_cells = [nn.GroupNorm] if dtype == ms.float16 else [nn.AvgPool2d, TrilinearInterpolate]
        vae = auto_mixed_precision(vae, amp_level, dtype, custom_fp32_cells=custom_fp32_cells)
        logger.info(f"Set mixed precision to O2 with dtype={args.precision}")
    elif args.precision == "fp32":
        amp_level = "O0"
    else:
        raise ValueError(f"Unsupported precision {args.precision}")

    ds_config = dict(
        data_file_path=args.data_file_path,
        video_column=args.video_column,
        data_folder=real_video_dir,
        size=resolution,
        crop_size=crop_size,
        disable_flip=True,
        random_crop=False,
    )
    if args.dataset_name == "video":
        ds_config.update(
            dict(
                sample_stride=sample_rate,
                sample_n_frames=num_frames,
                return_image=False,
                dynamic_start_index=args.dynamic_start_index,
            )
        )
        split_time_upsample = True
        assert not (
            args.num_frames % 2 == 0 and split_time_upsample
        ), "num of frames must be odd if split_time_upsample is True"
    else:
        ds_config.update(dict(expand_dim_t=args.expand_dim_t))
    dataset = VideoDataset(**ds_config)
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        ds_name=args.dataset_name,
        num_parallel_workers=num_workers,
        shuffle=False,  # be in order
        device_num=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
        rank_id=rank_id if not get_sequence_parallel_state() else hccl_info.group_id,
        drop_remainder=False,
    )
    num_batches = dataloader.get_dataset_size()
    logger.info("Number of batches: %d", num_batches)
    ds_iter = dataloader.create_dict_iterator(1)
    # ---- Prepare Dataset

    # ---- Inference ----
    for batch in tqdm(ds_iter, total=num_batches):
        if args.dataset_name == "image":
            x = batch["image"]
        else:
            x = batch["video"]
        file_paths = batch["path"]
        x = x.to(dtype=dtype)  # b c t h w
        latents = vae.encode(x)
        video_recon = vae.decode(latents)
        for idx, video in enumerate(video_recon):
            file_name = os.path.basename(eval(str(file_paths))[idx])
            if ".avi" in os.path.basename(file_name):
                file_name = file_name.replace(".avi", ".mp4")
            output_path = os.path.join(generated_video_dir, file_name)
            if not os.path.exists(os.path.dirname(output_path)):
                os.mkdir(os.path.dirname(output_path))
            if args.output_origin:
                os.makedirs(os.path.join(generated_video_dir, "origin/"), exist_ok=True)
                origin_output_path = os.path.join(generated_video_dir, "origin/", file_name)
                save_data = transform_to_rgb(x[idx : idx + 1].to(ms.float32).asnumpy(), rescale_to_uint8=False)
                # (b c t h w) -> (b t h w c)
                save_data = np.transpose(save_data, (0, 2, 3, 4, 1))
                if not os.path.exists(os.path.dirname(origin_output_path)):
                    os.mkdir(os.path.dirname(origin_output_path))
                save_videos(
                    save_data,
                    origin_output_path,
                    loop=0,
                    fps=sample_fps / sample_rate,
                )
            video = video.unsqueeze(0)  # (bs=1)
            save_data = transform_to_rgb(video.to(ms.float32).asnumpy(), rescale_to_uint8=False)
            # (b t c h w) -> (b t h w c)
            save_data = np.transpose(save_data, (0, 1, 3, 4, 2))
            save_videos(
                save_data,
                output_path,
                loop=0,
                fps=sample_fps / sample_rate,
            )
    logger.info(f"Finish video reconstruction, and save videos to {generated_video_dir}")
    # ---- Inference ----


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--real_video_dir", type=str, default="")
    parser.add_argument("--generated_video_dir", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="results/pretrained/causal_vae.ckpt")
    parser.add_argument(
        "--model_config",
        default="scripts/causalvae/release.json",
        help="the model configuration file for the causalvae.",
    )
    parser.add_argument("--sample_fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument(
        "--expand_dim_t",
        default=False,
        type=str2bool,
        help="expand temporal axis for image data, used for vae 3d inference with image data",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--output_origin", action="store_true")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        help="mixed precision type, if fp32, all layer precision is float32 (amp_level=O0),  \
                if bf16 or fp16, amp_level==O2, part of layers will compute in bf16 or fp16 such as matmul, dense, conv.",
    )
    parser.add_argument("--device", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--parallel_mode", default="data", type=str, choices=["data", "optim"], help="parallel mode: data, optim"
    )
    parser.add_argument("--jit_level", default="O0", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument(
        "--dataset_name", default="video", type=str, choices=["image", "video"], help="dataset name, image or video"
    )
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--dynamic_start_index",
        action="store_true",
        help="Whether to use a random frame as the starting frame for reconstruction. Default is False for the ease of evaluation.",
    )
    parser.add_argument(
        "--data_file_path",
        default=None,
        help="The data file path where the video paths are recorded. Now support json and csv file"
        "If not provided, will search all videos under `video_path` in a recursive manner.",
    )
    parser.add_argument(
        "--video_column",
        default="video",
        help="The column of video file path in `data_file_path`. Defaults to `video`.",
    )
    args = parser.parse_args()
    main(args)
