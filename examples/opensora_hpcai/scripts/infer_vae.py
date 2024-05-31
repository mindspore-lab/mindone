import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import mindspore as ms
from mindspore import nn
from mindspore.communication.management import get_group_size, get_rank, init

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from opensora.datasets.video_dataset import create_dataloader
from opensora.models.vae.vae import SD_CONFIG, AutoencoderKL
from opensora.utils.amp import auto_mixed_precision
from opensora.utils.model_utils import str2bool  # _check_cfgs_in_parser

from mindone.utils.logger import set_logger
from mindone.utils.misc import to_abspath
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    enable_dvm: bool = False,
    global_bf16: bool = False,
):
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
        # FIXME: the graph_kernel_flags settting is a temp solution to fix dvm loss convergence in ms2.3-rc2. Refine it for future ms version.
        ms.set_context(enable_graph_kernel=True, graph_kernel_flags="--disable_cluster_ops=Pow,Select")

    if global_bf16:
        ms.set_context(ascend_config={"precision_mode": "allow_mix_precision_bf16"})

    return rank_id, device_num


def main(args):
    set_logger(name="", output_dir="logs/infer_vae")
    rank_id, device_num = init_env(
        args.mode, args.seed, args.use_parallel, device_target=args.device_target, global_bf16=args.global_bf16
    )
    print(f"rank_id {rank_id}, device_num {device_num}")

    # build dataloader for large amount of captions
    ds_config = dict(
        csv_path=args.csv_path,
        video_folder=args.video_folder,
        sample_size=args.image_size,
        sample_stride=args.frame_stride,
        micro_batch_size=args.vae_micro_batch_size,
        video_column=args.video_column,
        caption_column=args.caption_column,
        return_frame_data=args.dl_return_all_frames,
        transform_name=args.transform_name,
    )
    dataloader, ds = create_dataloader(
        ds_config,
        args.batch_size,
        ds_name="video",
        num_parallel_workers=16,
        max_rowsize=256,
        shuffle=False,  # be in order
        device_num=device_num,
        rank_id=rank_id,
        drop_remainder=False,
        return_dataset=True,
    )
    dataset_size = dataloader.get_dataset_size()
    logger.info(f"Num batches: {dataset_size}")

    # model initiate and weight loading
    logger.info("vae init")
    VAE_Z_CH = SD_CONFIG["z_channels"]
    vae = AutoencoderKL(
        SD_CONFIG,
        VAE_Z_CH,
        ckpt_path=args.vae_checkpoint,
    )
    vae = vae.set_train(False)
    dtype_map = {"fp16": ms.float16, "bf16": ms.bfloat16}
    if args.dtype in ["fp16", "bf16"]:
        vae = auto_mixed_precision(
            vae,
            amp_level=args.amp_level,
            dtype=dtype_map[args.vae_dtype],
            custom_fp32_cells=[nn.GroupNorm],
        )

    logger.info("Start VAE embedding...")

    def save_output(video_name, mean, std=None, fps=None, ori_size=None):
        fn = Path(str(video_name)).with_suffix(".npz")
        npz_fp = os.path.join(output_folder, fn)
        if not os.path.exists(os.path.dirname(npz_fp)):
            os.makedirs(os.path.dirname(npz_fp))
        if os.path.exists(npz_fp):
            if args.allow_overwrite:
                logger.info(f"Overwritting {npz_fp}")
        if args.save_distribution:
            np.savez(
                npz_fp,
                latent_mean=mean.astype(np.float32),
                latent_std=std.astype(np.float32),
                fps=fps,
                ori_size=ori_size,
            )
        else:
            np.savez(
                npz_fp,
                latent_mean=video_latent_mean.astype(np.float32),
                fps=fps,
                ori_size=ori_size,
            )
        return npz_fp

    # infer
    if args.csv_path is not None:
        if args.output_path in [None, ""]:
            output_folder = os.path.dirname(args.csv_path)
        else:
            output_folder = args.output_path
        os.makedirs(output_folder, exist_ok=True)

        logger.info(f"Output embeddings will be saved: {output_folder}")

        ds_iter = dataloader.create_dict_iterator(1, output_numpy=True)
        for step, data in tqdm(enumerate(ds_iter), total=dataset_size):
            start_time = time.time()
            # caption = data["caption"]

            if args.dl_return_all_frames:
                frame_data = data["frame_data"]
                num_videos = frame_data.shape[0]
                fps = data["fps"][0]
                ori_size = data['ori_size'][0]
                assert args.batch_size == 1, "batch size > 1 is not supported due to dynamic frame numbers among videos"
                for i in range(num_videos):
                    video_path = data["video_path"][i]

                    fn = Path(str(video_path)).with_suffix(".npz")
                    npz_fp = os.path.join(output_folder, fn)
                    if os.path.exists(npz_fp) and not args.allow_overwrite:
                        logger.info(f"{npz_fp} exists, skip vae encoding")
                        continue

                    video_latent_mean = []
                    video_latent_std = []

                    x = frame_data[i]
                    bs = args.vae_micro_batch_size
                    for j in range(0, x.shape[0], bs):
                        x_bs = x[j : min(j + bs, x.shape[0])]
                        mean, std = ms.ops.stop_gradient(vae.encode_with_moments_output(ms.Tensor(x_bs, ms.float32)))
                        video_latent_mean.append(mean.asnumpy())
                        if args.save_distribution:
                            video_latent_std.append(std.asnumpy())

                    video_latent_mean = np.concatenate(video_latent_mean, axis=0)
                    if args.save_distribution:
                        video_latent_std = np.concatenate(video_latent_std, axis=0)

                    save_output(video_path, video_latent_mean, video_latent_std, fps, ori_size)
            else:
                num_videos = data["video_path"].shape[0]
                for i in range(num_videos):
                    video_path = data["video_path"][i]
                    abs_video_path = os.path.join(args.video_folder, video_path)

                    fn = Path(str(video_path)).with_suffix(".npz")
                    npz_fp = os.path.join(output_folder, fn)
                    if os.path.exists(npz_fp) and not args.allow_overwrite:
                        logger.info(f"{npz_fp} exists, skip vae encoding")
                        continue

                    video_latent_mean = []
                    video_latent_std = []
                    fps, ori_size = None, None
                    for x_bs, fps, ori_size  in ds.get_video_frames_in_batch(
                        abs_video_path, micro_batch_size=args.vae_micro_batch_size, sample_stride=args.frame_stride
                    ):
                        mean, std = ms.ops.stop_gradient(vae.encode_with_moments_output(ms.Tensor(x_bs, ms.float32)))
                        video_latent_mean.append(mean.asnumpy())
                        if args.save_distribution:
                            video_latent_std.append(std.asnumpy())
                        fps = fps
                        ori_size = ori_size

                    video_latent_mean = np.concatenate(video_latent_mean, axis=0)
                    if args.save_distribution:
                        video_latent_std = np.concatenate(video_latent_std, axis=0)

                    save_output(video_path, video_latent_mean, video_latent_std, fps, ori_size)

            end_time = time.time()
            logger.info(f"Time cost: {end_time-start_time:0.3f}s")
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
        "--output_path",
        type=str,
        default=None,
        help="output dir to save the embeddings, if None, will treat the parent dir of csv_path as output_path.",
    )
    parser.add_argument(
        "--save_distribution",
        default=True,
        type=str2bool,
        help="If True, will save mean and std representing vae latent distribution. \
                Otherwise, will only save mean (save half storage but loss vae sampling diversity).",
    )
    parser.add_argument("--video_column", default="video", type=str, help="name of column for videos saved in csv file")
    parser.add_argument(
        "--caption_column", default="caption", type=str, help="name of column for captions saved in csv file"
    )
    parser.add_argument("--video_folder", default="", type=str, help="root dir for the video data")
    parser.add_argument("--image_size", nargs="+", default=[512, 512], type=int, help="image size")
    parser.add_argument(
        "--dl_return_all_frames",
        default=True,
        type=str2bool,
        help="dataloder return all frames. If True, read all frames in a video then do vae encoding in micro_batch_size (faster but cost more CPU memory). \
                If False, read a clip of frames in a video in micro_batch_size and do vae encoding iteratively. (slower but memory efficient",
    )
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
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
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
    parser.add_argument("--frame_stride", default=1, type=int, help="frame sampling stride")
    parser.add_argument("--transform_name", default='center', type=str, help="center or crop_resize, if center, resize by the short side to h then center crop. If crop_resize, center crop maximally according to the AR of target image size then resize, suitable for where target h != target w.")
    parser.add_argument(
        "--vae_micro_batch_size",
        type=int,
        default=64,
        help="If not None, split batch_size*num_frames into smaller ones for VAE encoding to reduce memory limitation",
    )
    parser.add_argument(
        "--use_recompute",
        default=False,
        type=str2bool,
        help="whether use recompute.",
    )
    parser.add_argument(
        "--global_bf16",
        default=False,
        type=str2bool,
        help="Experimental. If True, dtype will be overrided, operators will be computered in bf16 if they are supported by CANN",
    )
    parser.add_argument(
        "--allow_overwrite",
        type=str2bool,
        default=False,
        help="If True, allow to overwrite the existing npz file. If False, will skip vae encoding if the latent npz file is already existed",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")

    default_args = parser.parse_args()
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(__dir__, ".."))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = to_abspath(abs_path, default_args.config)
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
    # convert to absolute path, necessary for modelarts
    args.csv_path = to_abspath(abs_path, args.csv_path)
    args.output_path = to_abspath(abs_path, args.output_path)
    args.video_folder = to_abspath(abs_path, args.video_folder)
    args.vae_checkpoint = to_abspath(abs_path, args.vae_checkpoint)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
