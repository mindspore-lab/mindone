import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from opensora.data.t2v_dataset import create_dataloader
from opensora.models.autoencoder import SD_CONFIG, AutoencoderKL
from opensora.utils.model_utils import str2bool  # _check_cfgs_in_parser
from tqdm import tqdm

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    rank_id = 0
    ms.set_context(
        mode=args.mode,
        device_target=args.device_target,
        device_id=device_id,
    )
    if args.precision_mode is not None:
        ms.set_context(ascend_config={"precision_mode": args.precision_mode})

    device_num = 1
    return device_id, rank_id, device_num


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    device_id, rank_id, device_num = init_env(args)
    print(f"D--: rank_id {rank_id}, device_num {device_num}")
    set_random_seed(args.seed)

    # build dataloader for large amount of captions
    if args.csv_path is not None:
        ds_config = dict(
            csv_path=args.csv_path,
            video_folder=args.video_folder,
            return_text_emb=False,
            sample_size=args.image_size,
            sample_stride=1,
            sample_n_frames=16,
            tokenizer=None,
            video_column=args.video_column,
            caption_column=args.caption_column,
            disable_flip=True,
        )
        _, dataset = create_dataloader(
            ds_config,
            args.batch_size,
            ds_name="text_video",
            num_parallel_workers=12,
            max_rowsize=32,
            shuffle=False,  # be in order
            device_num=device_num,
            rank_id=rank_id,
            drop_remainder=False,
            return_dataset=True,
        )
        logger.info(f"Num samples: {dataset}")

    # model initiate and weight loading
    logger.info("vae init")

    VAE_Z_CH = SD_CONFIG["z_channels"]
    vae = AutoencoderKL(
        SD_CONFIG,
        VAE_Z_CH,
        ckpt_path=args.vae_checkpoint,
        use_fp16=False,
    )
    vae = vae.set_train(False)

    logger.info("Start VAE embedding...")

    # infer
    if args.csv_path is not None:
        if args.output_dir is None:
            output_folder = os.path.dirname(args.csv_path)
        else:
            output_folder = args.output_dir
        os.makedirs(output_folder, exist_ok=True)

        logger.info(f"Output embeddings will be saved: {output_folder}")

        for video_index in tqdm(range(len(dataset)), total=len(dataset)):
            start_time = time.time()
            video_latent_mean = []
            video_latent_std = []
            for video_name, _, inputs in dataset.traverse_single_video_frames(video_index):
                video_data = inputs["video"]
                latent_momentum = ms.ops.stop_gradient(
                    vae.encode_with_moments_output(ms.Tensor(video_data, ms.float32))
                )
                mean, std = latent_momentum.chunk(2, axis=1)
                video_latent_mean.append(mean.asnumpy())
                video_latent_std.append(std.asnumpy())
            video_latent_mean = np.concatenate(video_latent_mean, axis=0)
            video_latent_std = np.concatenate(video_latent_std, axis=0)

            end_time = time.time()
            logger.info(f"Time cost: {end_time-start_time:0.3f}s for video {video_name}")

            fn = Path(str(video_name)).with_suffix(".npz")
            npz_fp = os.path.join(output_folder, fn)
            if not os.path.exists(os.path.dirname(npz_fp)):
                os.makedirs(os.path.dirname(npz_fp))
            if os.path.exists(npz_fp):
                if args.allow_overwrite:
                    logger.info(f"Overwritting {npz_fp}")
                else:
                    raise ValueError(f"{npz_fp} already exist!")
            np.savez(
                npz_fp,
                latent_mean=video_latent_mean.astype(np.float32),
                latent_std=video_latent_std.astype(np.float32),
            )
        logger.info(f"Done. Embeddings saved in {output_folder}")

    else:
        raise ValueError("Must provide csv file!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments. It can contain captions.",
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        type=str,
        help="path to csv annotation file, If None, video_caption.csv is expected to live under `data_path`",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output dir to save the embeddings, if None, will treat the parent dir of csv_path as output dir.",
    )
    parser.add_argument("--video_column", default="video", type=str, help="name of column for videos saved in csv file")
    parser.add_argument(
        "--caption_column", default="caption", type=str, help="name of column for captions saved in csv file"
    )
    parser.add_argument("--video_folder", default="", type=str, help="root dir for the video data")
    parser.add_argument("--image_size", default=256, type=int, help="image size")
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-ema.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )

    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp32`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument(
        "--use_recompute",
        default=False,
        type=str2bool,
        help="whether use recompute.",
    )
    parser.add_argument("--allow_overwrite", action="store_true", help="allow to overwrite the existing npz file.")

    parser.add_argument("--batch_size", default=8, type=int, help="batch size")

    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))

    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            # _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(
                **dict(
                    captions=cfg["captions"],
                    t5_model_dir=cfg["t5_model_dir"],
                )
            )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
