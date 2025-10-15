import argparse
import logging
import os
import sys

import numpy as np
from tqdm import tqdm

from mindspore import GRAPH_MODE, Tensor
from mindspore import dtype as mstype
from mindspore import get_context, set_context

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger

sys.path.append(".")

from hyvideo.constants import PRECISION_TO_TYPE, PRECISIONS, VAE_PATH
from hyvideo.dataset.vae_dataset import VideoDataset
from hyvideo.utils.ms_utils import init_env
from hyvideo.vae import load_vae

from mindone.data import create_dataloader

logger = logging.getLogger(__name__)


def to_numpy(x: Tensor) -> np.ndarray:
    if x.dtype == mstype.bfloat16:
        x = x.astype(mstype.float32)
    return x.asnumpy()


def vae_latent_cache(vae, input, dtype=None):
    dtype = input.dtype if dtype is None else dtype
    mean, logvar = vae._encode(input)
    mean, logvar = to_numpy(mean), to_numpy(logvar)  # b c t h w
    std = np.exp(0.5 * np.clip(logvar, -30.0, 20.0))
    return mean, std


def process_folder(args, vae, dtype, rank_id, device_num):
    input_video_dir = args.input_video_dir
    latent_cache_dir = args.latent_cache_dir
    height, width = args.height, args.width
    num_frames = 10**5  # read the full video, limitation of `albumentations` (i.e., `additional_targets`)
    sample_rate = args.sample_rate

    batch_size = 1  # dynamic num_frames , use bs=1
    num_workers = args.num_workers

    if not os.path.exists(args.latent_cache_dir):
        os.makedirs(args.latent_cache_dir, exist_ok=True)

    ds_config = dict(
        data_file_path=args.data_file_path,
        video_column=args.video_column,
        data_folder=input_video_dir,
        size=(height, width),
        crop_size=(height, width),
        disable_flip=True,
        random_crop=False,
    )

    ds_config.update(
        dict(
            sample_stride=sample_rate,
            sample_n_frames=num_frames,
            return_image=False,
            dynamic_start_index=False,
            frames_duplications=False,
        )
    )

    dataset = VideoDataset(**ds_config)
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # be in order
        device_num=device_num,
        rank_id=rank_id,
        drop_remainder=False,
    )
    num_batches = dataloader.get_dataset_size()
    logger.info("Number of batches: %d", num_batches)
    ds_iter = dataloader.create_dict_iterator(1)
    vae_compression_factor = 4
    for batch in tqdm(ds_iter, total=num_batches):
        x = batch["video"]
        file_paths = batch["path"]
        x = x.to(dtype=dtype)  # b c t h w
        num_frames = x.shape[2]
        vae_num_frames = (num_frames // vae_compression_factor) * vae_compression_factor + 1
        if vae_num_frames > num_frames:
            vae_num_frames = vae_num_frames - vae_compression_factor

        x = x[:, :, :vae_num_frames]
        batch_mean, batch_std = vae_latent_cache(vae, x, dtype=dtype)
        idx = 0
        for mean, std in zip(batch_mean, batch_std):
            file_paths = eval(str(file_paths).replace("/n", ","))
            file_path = file_paths[idx]
            # change the file name to .npz
            ori_basename = os.path.basename(file_path)
            new_basename = ori_basename.split(".")[0] + ".npz"
            file_path = file_path.replace(ori_basename, new_basename)
            # change the file path to the same folder structure as the input video folder
            input_video_dir = input_video_dir.rstrip(os.sep)
            latent_cache_dir = latent_cache_dir.rstrip(os.sep)
            file_path = file_path.rstrip(os.sep)
            output_path = file_path.replace(input_video_dir, latent_cache_dir)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez(output_path, latent_mean=mean, latent_std=std)
            idx += 1

    logger.info(f"Finish latent caching, and save cache to {latent_cache_dir}")


def main(args):
    rank_id, device_num = init_env(
        mode=args.mode,
        device_target=args.device,
        precision_mode=args.precision_mode,
        jit_level=args.jit_level,
        jit_syntax_level=args.jit_syntax_level,
        seed=args.seed,
        distributed=args.use_parallel,
    )
    mode = get_context("mode")
    # if graph mode and vae tiling is ON, uise dfs exec order
    if mode == GRAPH_MODE and args.vae_tiling:
        set_context(exec_order="dfs")
    set_logger(name="", output_dir=".", rank=0)

    vae, _, s_ratio, t_ratio = load_vae(
        args.vae,
        logger=logger,
        precision=args.vae_precision,
        checkpoint=args.ms_checkpoint,
        tiling=args.vae_tiling,
    )
    dtype = PRECISION_TO_TYPE[args.vae_precision]

    process_folder(args, vae, dtype, rank_id, device_num)


def get_parser():
    parser = argparse.ArgumentParser()

    # Video Group
    parser.add_argument(
        "--height", type=int, default=336, help="Height of the processed video frames. It applies to image size too."
    )
    parser.add_argument(
        "--width", type=int, default=336, help="Width of the processed video frames. It applies to image size too."
    )

    parser.add_argument("--sample-rate", type=int, default=1, help="Sampling rate for video frames.")

    # Video Folder Group
    parser.add_argument(
        "--input-video-dir", type=str, default="", help="Directory containing input videos for processing."
    )
    parser.add_argument("--latent-cache-dir", type=str, default="", help="Directory to save latent cache.")
    # parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument(
        "--data-file-path",
        default=None,
        help="Data file path where video paths are recorded. Supports json and csv files. "
        "If not provided, will search all videos under `input_video_dir` recursively.",
    )
    parser.add_argument(
        "--video-column", type=str, default="video", help="The video column name in the provided Data file path."
    )

    # Other Group
    parser.add_argument(
        "--vae", type=str, default="884-16c-hy", choices=list(VAE_PATH), help="Name of the VAE model to use."
    )
    parser.add_argument(
        "--vae-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the VAE model: fp16, bf16, or fp32.",
    )
    parser.add_argument("--vae-tiling", action="store_true", help="Enable tiling for the VAE model to save GPU memory.")
    parser.add_argument("--ms-checkpoint", type=str, default=None, help="Path to the MindSpore checkpoint file.")

    # MindSpore setting
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode.")
    parser.add_argument("--device", type=str, default="Ascend", help="Device to run the model on: Ascend or GPU.")
    parser.add_argument(
        "--max-device-memory",
        type=str,
        default=None,
        help="Maximum device memory to use, e.g., `30GB` for 910a, `59GB` for 910b.",
    )
    parser.add_argument("--use-parallel", default=False, type=str2bool, help="Use parallel processing.")
    parser.add_argument(
        "--parallel-mode", default="data", type=str, choices=["data", "optim"], help="Parallel mode: data or optim."
    )
    parser.add_argument("--jit-level", default="O0", help="Set JIT level: O0: KBK, O1: DVM, O2: GE.")
    parser.add_argument(
        "--jit-syntax-level", default="strict", choices=["strict", "lax"], help="Set JIT syntax level: strict or lax."
    )
    parser.add_argument("--seed", type=int, default=4, help="Random seed for inference.")
    parser.add_argument(
        "--precision-mode", default=None, type=str, help="Set precision mode for Ascend configurations."
    )
    parser.add_argument(
        "--vae-keep-gn-fp32",
        default=False,
        type=str2bool,
        help="Keep GroupNorm in fp32. Defaults to False in inference, better to set to True when training VAE.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)
