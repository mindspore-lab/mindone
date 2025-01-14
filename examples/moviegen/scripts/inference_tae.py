import glob
import logging
import os
import sys
from math import ceil
from pathlib import Path
from typing import List, Optional

import numpy as np
from jsonargparse import ArgumentParser
from jsonargparse.typing import path_type
from tqdm import tqdm, trange

from mindspore import Tensor, amp
from mindspore import dtype as mstype
from mindspore import get_context, nn

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.append(mindone_lib_path)
sys.path.append(os.path.join(__dir__, ".."))

from mg.dataset.tae_dataset import VideoDataset
from mg.models.tae import TemporalAutoencoder
from mg.utils import to_numpy

from mindone.data import create_dataloader
from mindone.utils import init_train_env, set_logger
from mindone.visualize import save_videos

logger = logging.getLogger(__name__)

Path_dr = path_type("dr", docstring="path to a directory that exists and is readable")


def encode(args, tae: TemporalAutoencoder, save_dir: Path, rank_id: int, device_num: int, mode: int):
    dataset = VideoDataset(
        **args.video_data.init_args,
        sample_n_frames=10**5,  # read the full video, limitation of `albumentations` (i.e., `additional_targets`)
        output_columns=["video", "rel_path"],
    )
    dataloader = create_dataloader(
        dataset, drop_remainder=False, device_num=device_num, rank_id=rank_id, **args.dataloader
    )

    # print key info
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {mode}",
            f"Debug mode: {args.env.debug}",
            f"TAE dtype: {args.tae.dtype}",
            f"Image size: {args.video_data.init_args.size}",
            f"Crop size: {args.video_data.init_args.crop_size}",
            f"Num of batches: {dataloader.get_dataset_size()}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    for samples in tqdm(dataloader.create_tuple_iterator(num_epochs=1), total=dataloader.get_dataset_size()):
        _, mean, logvar = tae.encode(samples[0])
        mean, logvar = to_numpy(mean), to_numpy(logvar)
        std = np.exp(0.5 * np.clip(logvar, -30.0, 20.0))

        # B C T H W -> B T C H W
        mean = np.transpose(mean, (0, 2, 1, 3, 4))
        std = np.transpose(std, (0, 2, 1, 3, 4))

        for m, s, path in zip(mean, std, samples[1].tolist()):
            out_path = save_dir / path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path.with_suffix(".npz"), latent_mean=m, latent_std=s)
    logger.info(f"Completed. Latents saved in {save_dir}")


def prepare_latents(folder: Optional[Path_dr] = None, rank_id: int = 0, device_num: int = 1) -> List[str]:
    latents = sorted(glob.glob(os.path.join(folder, "*.npy")))
    latents = latents[rank_id::device_num]
    logger.info(f"Number of latents for rank {rank_id}: {len(latents)}")
    return latents


def decode(args, tae: TemporalAutoencoder, save_dir: Path, rank_id: int, device_num: int, mode: int):
    latent_paths = prepare_latents(**args.latent_data, rank_id=rank_id, device_num=device_num)
    batch_size = args.dataloader.batch_size

    # print key info
    latent_shape = np.load(latent_paths[0]).shape
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {mode}",
            f"Debug mode: {args.env.debug}",
            f"TAE dtype: {args.tae.dtype}",
            f"Latent shape: {latent_shape}",
            f"Num of batches: {ceil(len(latent_paths) / batch_size)}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    for i in trange(0, len(latent_paths), batch_size):
        lps = latent_paths[i : i + batch_size]
        latents = np.stack([np.load(lp) for lp in lps])

        latents = np.transpose(latents, (0, 2, 1, 3, 4))  # FIXME: remove this redundancy
        latents = latents / tae.scale_factor + tae.shift_factor
        videos = to_numpy(tae.decode(Tensor(latents), target_num_frames=args.num_frames))
        videos = np.clip((videos + 1.0) / 2.0, 0.0, 1.0)
        videos = np.transpose(videos, (0, 2, 3, 4, 1))

        for lp, video in zip(lps, videos):
            save_fp = save_dir / Path(lp).with_suffix("." + args.save_format).name
            save_videos(video, str(save_fp), fps=args.fps)
    logger.info(f"Completed. Videos saved in {save_dir}")


def main(args):
    # 1. init env
    _, rank_id, device_num = init_train_env(**args.env)  # TODO: rename as train and infer are identical?
    mode = get_context("mode")  # `init_train_env()` may change the mode during debugging

    save_dir = Path(args.output_path.absolute)
    save_dir.mkdir(parents=True, exist_ok=True)
    set_logger(name="", output_dir=str(save_dir), rank=rank_id)

    # 3. TAE initiate and weight loading
    logger.info("Initializing TAE...")
    tae = TemporalAutoencoder(**args.tae).set_train(False)
    if tae.dtype != mstype.float32:
        # FIXME: remove AMP and add custom dtype conversion support for better compatibility with PyNative
        amp.custom_mixed_precision(
            tae, black_list=amp.get_black_list() + [nn.GroupNorm, nn.AvgPool2d, nn.Upsample], dtype=tae.dtype
        )
    # TODO: add dynamic shape support

    if args.video_data is not None:
        logger.info("Encoding video data.")
        encode(args, tae, save_dir, rank_id, device_num, mode)
    elif args.latent_data.folder is not None:
        logger.info("Decoding latent data.")
        decode(args, tae, save_dir, rank_id, device_num, mode)
    else:
        raise ValueError("Either `video_data` or `latent_data` must be provided.")


if __name__ == "__main__":
    parser = ArgumentParser(description="TAE inference script.")
    parser.add_function_arguments(init_train_env, "env")
    parser.add_class_arguments(TemporalAutoencoder, "tae", instantiate=False)
    parser.add_subclass_arguments(
        VideoDataset,
        "video_data",
        skip={"random_crop", "flip", "sample_n_frames", "return_image", "output_columns"},
        instantiate=False,
        required=False,
    )
    parser.add_function_arguments(prepare_latents, "latent_data", skip={"rank_id", "device_num"})
    parser.add_function_arguments(
        create_dataloader,
        "dataloader",
        skip={
            "dataset",
            "transforms",
            "batch_transforms",
            "project_columns",
            "shuffle",
            "num_workers",  # no transformations inside `.map()`
            "drop_remainder",
            "device_num",
            "rank_id",
            "enable_modelarts",
        },
    )
    parser.link_arguments("env.debug", "dataloader.debug", apply_on="parse")
    parser.add_argument("--num_frames", default=256, type=int, help="Number of in the output video.")
    parser.add_argument("--fps", type=int, default=16, help="FPS in the saved video")
    parser.add_argument(
        "--save_format",
        default="mp4",
        choices=["gif", "mp4", "png"],
        type=str,
        help="video format for saving the sampling output: gif, mp4 or png",
    )
    parser.add_argument(
        "--output_path",
        default="output/",
        type=path_type("dcc"),  # path to a directory that can be created if it does not exist
        help="Output directory to save training results.",
    )
    cfg = parser.parse_args()
    main(cfg)
