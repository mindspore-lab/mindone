import argparse
import os
import sys

from tqdm import tqdm

sys.path.append(".")
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from opensora.models.causalvideovae.model import ModelRegistry
from opensora.models.causalvideovae.model.dataset_videobase import VideoDataset, create_dataloader
from opensora.npu_config import npu_config
from opensora.utils.utils import get_precision
from opensora.utils.video_utils import save_videos


def main(args: argparse.Namespace):
    rank_id, device_num = npu_config.set_npu_env(args)
    npu_config.print_ops_dtype_info()
    real_video_dir = args.real_video_dir
    generated_video_dir = args.generated_video_dir
    sample_rate = args.sample_rate
    height, width = args.resolution, args.resolution
    crop_size = (height, width) if args.crop_size is None else (args.crop_size, args.crop_size)
    num_frames = args.num_frames
    sample_rate = args.sample_rate

    sample_fps = args.sample_fps
    batch_size = args.batch_size
    num_workers = args.num_workers
    subset_size = args.subset_size

    if not os.path.exists(args.generated_video_dir):
        os.makedirs(args.generated_video_dir, exist_ok=True)

    data_type = get_precision(args.precision)

    # ---- Load Model ----
    model_cls = ModelRegistry.get_model(args.model_name)
    vae = model_cls.from_pretrained(args.from_pretrained)
    vae = vae.to(data_type)
    if args.enable_tiling:
        vae.enable_tiling()
        vae.tile_overlap_factor = args.tile_overlap_factor

    # ---- Prepare Dataset ----
    ds_config = dict(
        data_folder=real_video_dir,
        size=max(height, width),  # SmallestMaxSize
        crop_size=crop_size,
        disable_flip=True,
        random_crop=False,
        sample_stride=sample_rate,
        sample_n_frames=num_frames,
        dynamic_start_index=args.dynamic_start_index,
    )
    dataset = VideoDataset(**ds_config)
    if subset_size:
        indices = range(subset_size)
        dataset.dataset = [dataset.dataset[i] for i in indices]
        dataset.length = len(dataset)

    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        ds_name="video",
        num_parallel_workers=num_workers,
        shuffle=False,  # be in order
        device_num=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
        rank_id=rank_id if not get_sequence_parallel_state() else hccl_info.group_id,
        drop_remainder=False,
    )

    # ---- Inference ----
    for batch in tqdm(dataloader):
        x, file_names = batch["video"], batch["file_name"]
        x = x.to(dtype=data_type)  # b c t h w
        x = x * 2 - 1
        encode_result = vae.encode(x)
        if isinstance(encode_result, tuple):
            encode_result = encode_result[0]
        latents = encode_result.to(data_type)
        video_recon = vae.decode(latents)
        if isinstance(video_recon, tuple):
            video_recon = video_recon[0]
        for idx, video in enumerate(video_recon):
            output_path = os.path.join(generated_video_dir, file_names[idx])
            if args.output_origin:
                os.makedirs(os.path.join(generated_video_dir, "origin/"), exist_ok=True)
                origin_output_path = os.path.join(generated_video_dir, "origin/", file_names[idx])
                save_videos(x[idx], origin_output_path, loop=0, fps=sample_fps / sample_rate)

            save_videos(
                video,
                output_path,
                loop=0,
                fps=sample_fps / sample_rate,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_video_dir", type=str, default="")
    parser.add_argument("--generated_video_dir", type=str, default="")
    parser.add_argument("--from_pretrained", type=str, default="")
    parser.add_argument("--sample_fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--output_origin", action="store_true")
    parser.add_argument("--model_name", type=str, default=None, help="")
    parser.add_argument(
        "--dynamic_start_index",
        action="store_true",
        help="Whether to use a random frame as the starting frame for reconstruction. Default is False for the ease of evaluation.",
    )
    parser.add_argument("--mode", default=1, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
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
    parser.add_argument("--use_parallel", action="store_true", help="use parallel")
    parser.add_argument(
        "--parallel_mode", default="data", type=str, choices=["data", "optim"], help="parallel mode: data, optim"
    )
    parser.add_argument("--jit_level", default="O0", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    parser.add_argument(
        "--jit_syntax_level", default="strict", choices=["strict", "lax"], help="Set jit syntax level: strict or lax"
    )
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    args = parser.parse_args()
    main(args)
