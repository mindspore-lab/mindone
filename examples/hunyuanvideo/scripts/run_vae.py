import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import mindspore as ms
from mindspore import GRAPH_MODE, get_context, set_context

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger
from mindone.visualize.videos import save_videos

sys.path.append(".")

from hyvideo.constants import PRECISION_TO_TYPE, PRECISIONS, VAE_PATH
from hyvideo.dataset.vae_dataset import VideoDataset
from hyvideo.utils.data_utils import preprocess_image, preprocess_video, read_video, transform_to_rgb
from hyvideo.utils.ms_utils import init_env
from hyvideo.vae import load_vae

from mindone.data import create_dataloader

logger = logging.getLogger(__name__)


def vae_reconstruction(vae, input, dtype=None, sample_posterior=True):
    # input: b c t h w
    dtype = input.dtype if dtype is None else dtype
    latents = vae.encode(input, sample_posterior=sample_posterior)
    latents = latents.to(dtype)
    recon = vae.decode(latents)  # b c t h w
    return recon


def process_image(args, vae, dtype):
    image_path = args.image_pathx
    input_x = np.array(Image.open(image_path))  # (h w c)
    assert input_x.shape[2] == 3, f"Expect the input image has three channels, but got shape {input_x.shape}"
    x_vae = preprocess_image(input_x, args.height, args.width)  # use image as a single-frame video
    x_vae = ms.Tensor(x_vae, dtype).unsqueeze(0)  # b c t h w
    image_recon = vae_reconstruction(vae, x_vae, dtype=dtype)

    save_fp = os.path.join(args.output_path, args.rec_path)
    x = image_recon[0, :, 0, :, :].squeeze().asnumpy()
    x = transform_to_rgb(x).transpose(1, 2, 0)  # c h w -> h w c
    image = Image.fromarray(x)
    image.save(save_fp)
    logger.info(f"Save reconstructed data to {save_fp}")


def process_video(args, vae, dtype):
    x_vae = preprocess_video(read_video(args.video_path, args.num_frames, args.sample_rate), args.height, args.width)
    x_vae = ms.Tensor(x_vae, dtype).unsqueeze(0)  # b c t h w
    video_recon = vae_reconstruction(vae, x_vae, dtype=dtype)
    video_recon = video_recon.permute((0, 2, 1, 3, 4))  # b t c h w

    save_fp = os.path.join(args.output_path, args.rec_path)
    if ".avi" in os.path.basename(save_fp):
        save_fp = save_fp.replace(".avi", ".mp4")
    if video_recon.shape[1] == 1:
        x = video_recon[0, 0, :, :, :].squeeze().to(ms.float32).asnumpy()
        x = transform_to_rgb(x).transpose(1, 2, 0)  # c h w -> h w c
        image = Image.fromarray(x)
        save_fp = save_fp.replace("mp4", "jpg")
        image.save(save_fp)
    else:
        save_video_data = video_recon.transpose(0, 1, 3, 4, 2).to(ms.float32).asnumpy()  # (b t c h w) -> (b t h w c)
        save_video_data = transform_to_rgb(save_video_data, rescale_to_uint8=False)
        save_videos(save_video_data, save_fp, loop=0, fps=args.fps)
    logger.info(f"Save reconstructed data to {save_fp}")


def process_folder(args, vae, dtype, rank_id, device_num):
    real_video_dir = args.real_video_dir
    generated_video_dir = args.generated_video_dir
    height, width = args.height, args.width
    num_frames = args.num_frames
    sample_rate = args.sample_rate
    sample_fps = args.sample_fps
    batch_size = args.batch_size
    num_workers = args.num_workers

    if not os.path.exists(args.generated_video_dir):
        os.makedirs(args.generated_video_dir, exist_ok=True)

    ds_config = dict(
        data_file_path=args.data_file_path,
        video_column=args.video_column,
        data_folder=real_video_dir,
        size=(height, width),
        crop_size=(height, width),
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
        num_workers=num_workers,
        shuffle=False,  # be in order
        device_num=device_num,
        rank_id=rank_id,
        drop_remainder=False,
    )
    num_batches = dataloader.get_dataset_size()
    logger.info("Number of batches: %d", num_batches)
    ds_iter = dataloader.create_dict_iterator(1)
    for batch in tqdm(ds_iter, total=num_batches):
        if args.dataset_name == "image":
            x = batch["image"]
        else:
            x = batch["video"]
        file_paths = batch["path"]
        x = x.to(dtype=dtype)  # b c t h w
        video_recon = vae_reconstruction(vae, x, dtype=dtype)
        video_recon = video_recon.permute((0, 2, 1, 3, 4))  # b t c h w
        for idx, video in enumerate(video_recon):
            file_paths = eval(str(file_paths).replace("/n", ","))
            file_path = file_paths[idx]
            if ".avi" in os.path.basename(file_path):
                file_path = file_path.replace(".avi", ".mp4")
            real_video_dir = real_video_dir.rstrip(os.sep)
            generated_video_dir = generated_video_dir.rstrip(os.sep)
            file_path = file_path.rstrip(os.sep)
            output_path = file_path.replace(
                real_video_dir, generated_video_dir
            )  # the same folder structure as the real video folder
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            video = video.unsqueeze(0)  # (bs=1)
            save_data = transform_to_rgb(video.to(ms.float32).asnumpy(), rescale_to_uint8=False)
            save_data = np.transpose(save_data, (0, 1, 3, 4, 2))
            save_videos(
                save_data,
                output_path,
                loop=0,
                fps=sample_fps / sample_rate,
            )
    logger.info(f"Finish video reconstruction, and save videos to {generated_video_dir}")


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
    set_logger(name="", output_dir=args.output_path, rank=0)

    vae, _, s_ratio, t_ratio = load_vae(
        args.vae,
        logger=logger,
        precision=args.vae_precision,
        checkpoint=args.ms_checkpoint,
        tiling=args.vae_tiling,
    )
    dtype = PRECISION_TO_TYPE[args.vae_precision]

    if args.input_type == "image":
        assert device_num == 1, "Only support single-device inference given single input"
        process_image(args, vae, dtype)
    elif args.input_type == "video":
        assert device_num == 1, "Only support single-device inference given single input"
        process_video(args, vae, dtype)
    elif args.input_type == "folder":
        process_folder(args, vae, dtype, rank_id, device_num)
    else:
        raise ValueError("Unsupported input type. Please choose from 'image', 'video', or 'folder'.")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-type",
        type=str,
        default="video",
        choices=["image", "video", "folder"],
        help="Type of input data: image, video, or folder.",
    )
    parser.add_argument(
        "--output-path", type=str, default="save_samples/", help="Path to save the reconstructed video or image path."
    )
    # Image Group
    parser.add_argument("--image-path", type=str, default="", help="Path to the input image file")

    # Video Group
    parser.add_argument(
        "--height", type=int, default=336, help="Height of the processed video frames. It applies to image size too."
    )
    parser.add_argument(
        "--width", type=int, default=336, help="Width of the processed video frames. It applies to image size too."
    )
    parser.add_argument("--video-path", type=str, default="", help="Path to the input video file.")
    parser.add_argument(
        "--rec-path",
        type=str,
        default="",
        help="Path to save the reconstructed video/image path, relative to the given output path.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
    parser.add_argument("--num-frames", type=int, default=65, help="Number of frames to sample from the video.")
    parser.add_argument("--sample-rate", type=int, default=1, help="Sampling rate for video frames.")
    parser.add_argument("--sample-fps", type=int, default=30, help="Sampling frames per second for the video.")

    # Video Folder Group
    parser.add_argument(
        "--real-video-dir", type=str, default="", help="Directory containing real videos for processing."
    )
    parser.add_argument("--generated-video-dir", type=str, default="", help="Directory to save generated videos.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument(
        "--data-file-path",
        default=None,
        help="Data file path where video paths are recorded. Supports json and csv files. "
        "If not provided, will search all videos under `real_video_dir` recursively.",
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
    parser.add_argument(
        "--dynamic-start-index", action="store_true", help="Use dynamic start index for video sampling."
    )
    parser.add_argument("--expand-dim-t", default=False, type=str2bool, help="Expand dimension t for the dataset.")
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
    parser.add_argument(
        "--dataset-name", default="video", type=str, choices=["image", "video"], help="Dataset name: image or video."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)
