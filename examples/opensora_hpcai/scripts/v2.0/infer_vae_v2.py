import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from mindspore import amp
from mindspore import dtype as mstype
from mindspore import nn, tensor
from mindspore.ops import stop_gradient

from mindone.utils import init_env, set_logger
from mindone.utils.misc import to_abspath

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from opensora.datasets.video_dataset import create_dataloader

# from opensora.models.vae.vae import SD_CONFIG, AutoencoderKL
from opensora.models.hunyuan_vae import CausalVAE3D_HUNYUAN
from opensora.utils.model_utils import str2bool  # _check_cfgs_in_parser

logger = logging.getLogger(__name__)


def save_output(output_folder: Path, video_name: Path, mean: np.array, std: np.array, conds: Optional[np.array] = None):
    fn = video_name.with_suffix(".npz")
    npz_fp = os.path.join(output_folder, fn)
    if not os.path.exists(os.path.dirname(npz_fp)):
        os.makedirs(os.path.dirname(npz_fp))
    if os.path.exists(npz_fp):
        if args.allow_overwrite:
            logger.info(f"Overwritting {npz_fp}")
    if conds is not None:
        np.savez(
            npz_fp,
            latent_mean=mean.astype(np.float32),
            latent_std=std.astype(np.float32),
            cond=conds.astype(np.float32),
        )
    else:
        np.savez(
            npz_fp,
            latent_mean=mean.astype(np.float32),
            latent_std=std.astype(np.float32),
        )


def main(args):
    set_logger(name="", output_dir="logs/infer_vae")
    _, rank_id, device_num = init_env(mode=args.mode, seed=args.seed, distributed=args.use_parallel)
    print(f"rank_id {rank_id}, device_num {device_num}")

    if args.resize_by_max_value and args.batch_size != 1:
        raise ValueError(
            f"Batch size must be 1 when `resize_by_max_value=True`, but get `batch_size={args.batch_size}`."
        )

    if args.vae_micro_batch_size is None:
        args.vae_micro_batch_size = args.num_frames

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
        resize_by_max_value=args.resize_by_max_value,
        transform_name=args.transform_name,  # TODO: check whether align with original repo
        filter_data=args.filter_data,
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
    # with open(args.config, "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # ae_config = config["ae"]

    # temp solution, copied from configs/opensora-v2-0/train/image.py
    # FIXME: load config from yaml
    ae_config = dict(
        # type="hunyuan_vae",
        from_pretrained=args.vae_checkpoint,
        dtype=args.vae_precision,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        latent_channels=16,
        use_spatial_tiling=True,
        use_temporal_tiling=False,
    )
    # FIXME: load config from yaml
    cond_config = dict(
        is_causal_vae=True,
        condition_config=dict(
            t2v=1,
            i2v_head=5,  # train i2v (image as first frame) with weight 5
            i2v_loop=1,  # train image connection with weight 1
            i2v_tail=1,  # train i2v (image as last frame) with weight 1
        )
        if args.return_cond
        else None,
    )
    model_ae = CausalVAE3D_HUNYUAN(**ae_config).set_train(False)
    del model_ae.decoder

    dtype_map = {"fp32": mstype.float32, "fp16": mstype.float16, "bf16": mstype.bfloat16}
    dtype = dtype_map[args.vae_precision]
    if args.vae_precision in ["fp16", "bf16"]:
        amp.custom_mixed_precision(model_ae, black_list=amp.get_black_list() + [nn.GroupNorm], dtype=dtype)

    logger.info("Start VAE embedding...")

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

            if args.dl_return_all_frames:
                frame_data = data["frame_data"]
                num_videos = frame_data.shape[0]
                assert args.batch_size == 1, "batch size > 1 is not supported due to dynamic frame numbers among videos"
                for i in range(num_videos):
                    abs_video_path = data["video_path"][i]
                    video_path = Path(abs_video_path).relative_to(args.video_folder)
                    fn = video_path.with_suffix(".npz")
                    npz_fp = os.path.join(output_folder, fn)
                    if os.path.exists(npz_fp) and not args.allow_overwrite:
                        logger.info(f"{npz_fp} exists, skip vae encoding")
                        continue

                    video_latent_mean = []
                    video_latent_std = []
                    conds = []
                    x = frame_data[i]
                    x = np.expand_dims(
                        np.transpose(x, (1, 0, 2, 3)), axis=0
                    )  # [f, c, h, w] -> [b, c, f, h, w], b must be 1
                    bs = args.vae_micro_batch_size

                    if args.num_frames > 0:  # TODO
                        if x.shape[2] < args.num_frames:
                            raise ValueError(
                                f"Video {video_path} has {x.shape[2]} frames, but expected >= {args.num_frames} frames."
                            )
                        x = x[:, :, : args.num_frames, :, :]
                    else:
                        logger.warning(
                            f"Invalid args.num_frames value: {args.num_frames}. Using all frames in the video."
                        )

                    logger.info(f"The shape [b, c, f, h, w] of video for vae encoding: {x.shape}")

                    for j in range(0, x.shape[2], bs):
                        x_bs = x[:, :, j : min(j + bs, x.shape[2]), :, :]

                        if cond_config.get("condition_config", None) is not None:
                            raise NotImplementedError
                            # condition for i2v & v2v
                            # x_0, cond = ms.ops.stop_gradient(prepare_visual_condition(x_bs, cond_config["condition_config"], model_ae))
                            # # TODO: pack function
                            # # cond = pack(cond, patch_size=ae_config.get("patch_size", 2))  # FIXME: general config, not ae_config
                            # conds.append(cond.asnumpy())
                        else:
                            x_0, posterior = stop_gradient(model_ae.encode(tensor(x_bs, dtype), return_posterior=True))
                        video_latent_mean.append(posterior.mean.asnumpy())
                        video_latent_std.append(posterior.std.asnumpy())

                    video_latent_mean = np.concatenate(video_latent_mean, axis=2)
                    video_latent_std = np.concatenate(video_latent_std, axis=2)
                    conds = np.concatenate(conds, axis=2) if len(conds) > 0 else None
                    save_output(output_folder, video_path, video_latent_mean, video_latent_std, conds)
            else:
                num_videos = data["video_path"].shape[0]
                for i in range(num_videos):
                    abs_video_path = data["video_path"][i]
                    video_path = Path(abs_video_path).relative_to(args.video_folder)
                    fn = video_path.with_suffix(".npz")
                    npz_fp = os.path.join(output_folder, fn)
                    if os.path.exists(npz_fp) and not args.allow_overwrite:
                        logger.info(f"{npz_fp} exists, skip vae encoding")
                        continue

                    video_latent_mean = []
                    video_latent_std = []
                    conds = []
                    for x_bs, fps, ori_size in ds.get_video_frames_in_batch(
                        abs_video_path, micro_batch_size=args.vae_micro_batch_size, sample_stride=args.frame_stride
                    ):
                        x_bs = np.expand_dims(
                            np.transpose(x_bs, (1, 0, 2, 3)), axis=0
                        )  # [f, c, h, w] -> [b, c, f, h, w]

                        if cond_config.get("condition_config", None) is not None:
                            raise NotImplementedError
                            # condition for i2v & v2v
                            # x_0, cond = ms.ops.stop_gradient(prepare_visual_condition(x_bs, cond_config["condition_config"], model_ae))
                            # # TODO: pack function
                            # # cond = pack(cond, patch_size=ae_config.get("patch_size", 2))  # FIXME: general config, not ae_config
                            # conds.append(cond.asnumpy())
                        else:
                            x_0, posterior = stop_gradient(model_ae.encode(tensor(x_bs, dtype), return_posterior=True))
                        video_latent_mean.append(posterior.mean.asnumpy())
                        video_latent_std.append(posterior.std.asnumpy())

                    video_latent_mean = np.concatenate(video_latent_mean, axis=2)
                    video_latent_std = np.concatenate(video_latent_std, axis=2)
                    conds = np.concatenate(conds, axis=2) if len(conds) > 0 else None
                    save_output(output_folder, video_path, video_latent_mean, video_latent_std, conds)

            end_time = time.time()
            logger.info(f"Time cost: {end_time-start_time:0.3f}s")
        logger.info(f"Done. Embeddings saved in {output_folder}")

    else:
        raise ValueError("Must provide csv file!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="configs/opensora-v2-0/train/stage1_i2v.yaml",
        help="Path to load a config yaml file.",
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
    parser.add_argument("--video_column", default="video", type=str, help="name of column for videos saved in csv file")
    parser.add_argument(
        "--caption_column", default="caption", type=str, help="name of column for captions saved in csv file"
    )
    parser.add_argument("--video_folder", default="", type=str, help="root dir for the video data")
    parser.add_argument("--return_cond", default=False, type=str2bool, help="Whether return condition embedding or not")
    parser.add_argument("--filter_data", default=False, type=str2bool, help="Filter non-existing videos.")
    parser.add_argument("--image_size", nargs="+", default=[256, 256], type=int, help="image size")
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
        default="hpcai-tech/Open-Sora-v2/hunyuan_vae.safetensors",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    # parser.add_argument(
    #     "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    # )
    parser.add_argument(
        "--vae_precision",
        type=str,
        default="fp32",
        choices=["bf16", "fp16", "fp32"],
        help="Precision mode for the VAE model: fp16, bf16, or fp32.",
    )

    parser.add_argument("--mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    # parser.add_argument(
    #     "--dtype",
    #     default="fp32",
    #     type=str,
    #     choices=["bf16", "fp16", "fp32"],
    #     help="what data type to use for latte. Default is `fp32`, which corresponds to ms.float16",
    # )
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument("--frame_stride", default=1, type=int, help="frame sampling stride")
    parser.add_argument(
        "--transform_name",
        default="center",
        type=str,
        help="center or crop_resize, if center, resize by the short side to h \
                then center crop. If crop_resize, center crop maximally according to \
                the AR of target image size then resize, suitable for where target h != target w.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=129,
        help="Crop the video to num_frames frames.",
    )
    parser.add_argument(
        "--vae_micro_batch_size",
        type=int,
        default=None,
        help="If not None, split batch_size*num_frames into smaller ones for VAE encoding to reduce memory limitation",
    )

    parser.add_argument(
        "--use_recompute",
        default=False,
        type=str2bool,
        help="whether use recompute.",
    )
    parser.add_argument(
        "--allow_overwrite",
        type=str2bool,
        default=False,
        help="If True, allow to overwrite the existing npz file. If False, will skip vae encoding if the latent npz file is already existed",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--resize_by_max_value", default=False, type=str2bool, help="resize the image by max instead.")

    __dir__ = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(__dir__, ".."))
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
