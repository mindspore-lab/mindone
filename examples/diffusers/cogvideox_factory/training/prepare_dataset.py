#!/usr/bin/env python3

import argparse
import functools
import json
import pathlib
import queue
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer

import mindspore as ms
from mindspore import ops
from mindspore.dataset import GeneratorDataset, vision

from mindone.diffusers import AutoencoderKLCogVideoX
from mindone.diffusers.training_utils import init_distributed_device, is_master, set_seed
from mindone.diffusers.utils import export_to_video, get_logger
from mindone.transformers import T5EncoderModel

import decord  # isort:skip

from dataset import VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop  # isort:skip


decord.bridge.set_bridge("native")

logger = get_logger(__name__)

DTYPE_MAPPING = {
    "fp32": ms.float32,
    "fp16": ms.float16,
    "bf16": ms.bfloat16,
}


def check_height(x: Any) -> int:
    x = int(x)
    if x % 16 != 0:
        raise argparse.ArgumentTypeError(
            f"`--height_buckets` must be divisible by 16, but got {x} which does not fit criteria."
        )
    return x


def check_width(x: Any) -> int:
    x = int(x)
    if x % 16 != 0:
        raise argparse.ArgumentTypeError(
            f"`--width_buckets` must be divisible by 16, but got {x} which does not fit criteria."
        )
    return x


def check_frames(x: Any) -> int:
    x = int(x)
    if x % 4 != 0 and x % 4 != 1:
        raise argparse.ArgumentTypeError(
            f"`--frames_buckets` must be of form `4 * k` or `4 * k + 1`, but got {x} which does not fit criteria."
        )
    return x


def get_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Whether or not to distributed process",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="Hugging Face model ID to use for tokenizer, text encoder and VAE.",
    )
    parser.add_argument("--data_root", type=str, required=True, help="Path to where training data is located.")
    parser.add_argument(
        "--dataset_file", type=str, default=None, help="Path to CSV file containing metadata about training data."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="If using a CSV file via the `--dataset_file` argument, this should be the name of the column containing the captions. If using the folder structure format for data loading, this should be the name of the file containing line-separated captions (the file should be located in `--data_root`).",  # noqa: E501
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="If using a CSV file via the `--dataset_file` argument, this should be the name of the column containing the video paths. If using the folder structure format for data loading, this should be the name of the file containing line-separated video paths (the file should be located in `--data_root`).",  # noqa: E501
    )
    parser.add_argument(
        "--id_token",
        type=str,
        default=None,
        help="Identifier token appended to the start of each prompt if provided.",
    )
    parser.add_argument(
        "--height_buckets",
        nargs="+",
        type=check_height,
        default=[480],
    )
    parser.add_argument(
        "--width_buckets",
        nargs="+",
        type=check_width,
        default=[720],
    )
    parser.add_argument(
        "--frame_buckets",
        nargs="+",
        type=check_frames,
        default=[49],
    )
    parser.add_argument(
        "--random_flip",
        type=float,
        default=None,
        help="If random horizontal flip augmentation is to be used, this should be the flip probability.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--video_reshape_mode",
        type=str,
        default=None,
        help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'none']",
    )
    parser.add_argument(
        "--save_image_latents",
        action="store_true",
        help="Whether or not to encode and store image latents, which are required for image-to-video finetuning. The image latents are the first frame of input videos encoded with the VAE.",  # noqa: E501
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory where preprocessed videos/latents/embeddings will be saved.",
    )
    parser.add_argument("--max_num_frames", type=int, default=49, help="Maximum number of frames in output video.")
    parser.add_argument(
        "--max_sequence_length", type=int, default=226, help="Max sequence length of prompt embeddings."
    )
    parser.add_argument("--target_fps", type=int, default=8, help="Frame rate of output videos.")
    parser.add_argument(
        "--save_latents_and_embeddings",
        action="store_true",
        help="Whether to encode videos/captions to latents/embeddings and save them in pytorch serializable format.",
    )
    parser.add_argument(
        "--use_slicing",
        action="store_true",
        help="Whether to enable sliced encoding/decoding in the VAE. Only used if `--save_latents_and_embeddings` is also used.",
    )
    parser.add_argument(
        "--use_tiling",
        action="store_true",
        help="Whether to enable tiled encoding/decoding in the VAE. Only used if `--save_latents_and_embeddings` is also used.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Number of videos to process at once in the VAE.")
    parser.add_argument(
        "--num_decode_threads",
        type=int,
        default=0,
        help="Number of decoding threads for `decord` to use. The default `0` means to automatically determine required number of threads.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Data type to use when generating latents and prompt embeddings.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--num_artifact_workers", type=int, default=4, help="Number of worker threads for serializing artifacts."
    )
    return parser.parse_args()


def compute_prompt_embeddings(
    text_encoder: T5EncoderModel,
    text_input_ids: ms.Tensor,
    num_videos_per_prompt: int = 1,
    dtype: ms.Type = None,
):
    batch_size = text_input_ids.shape[0]
    prompt_embeds = text_encoder(text_input_ids)[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.tile((1, num_videos_per_prompt, 1))
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


to_pil_image = vision.ToPIL()


def save_image(image: ms.Tensor, path: pathlib.Path) -> None:
    image = to_pil_image(image)
    image.save(path)


def save_video(video: ms.Tensor, path: pathlib.Path, fps: int = 8) -> None:
    video = [to_pil_image(frame) for frame in video]
    export_to_video(video, path, fps=fps)


def save_prompt(prompt: str, path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(prompt)


def save_metadata(metadata: Dict[str, Any], path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps(metadata))


def serialize_artifacts(
    batch_size: int,
    fps: int,
    images_dir: Optional[pathlib.Path] = None,
    image_latents_dir: Optional[pathlib.Path] = None,
    videos_dir: Optional[pathlib.Path] = None,
    video_latents_dir: Optional[pathlib.Path] = None,
    prompts_dir: Optional[pathlib.Path] = None,
    prompt_embeds_dir: Optional[pathlib.Path] = None,
    images: Optional[ms.Tensor] = None,
    image_latents: Optional[ms.Tensor] = None,
    videos: Optional[ms.Tensor] = None,
    video_latents: Optional[ms.Tensor] = None,
    prompts: Optional[List[str]] = None,
    prompt_embeds: Optional[ms.Tensor] = None,
) -> None:
    num_frames, height, width = videos.shape[1], videos.shape[3], videos.shape[4]
    metadata = [{"num_frames": num_frames, "height": height, "width": width}]

    data_folder_mapper_list = [
        (images, images_dir, lambda img, path: save_image(img[0], path), "png"),
        (image_latents, image_latents_dir, lambda data, path: np.save(path, data), "npy"),
        (videos, videos_dir, functools.partial(save_video, fps=fps), "mp4"),
        (video_latents, video_latents_dir, lambda data, path: np.save(path, data), "npy"),
        (prompts, prompts_dir, save_prompt, "txt"),
        (prompt_embeds, prompt_embeds_dir, lambda data, path: np.save(path, data), "npy"),
        (metadata, videos_dir, save_metadata, "txt"),
    ]
    filenames = [uuid.uuid4() for _ in range(batch_size)]

    for data, folder, save_fn, extension in data_folder_mapper_list:
        if data is None:
            continue
        for slice, filename in zip(data, filenames):
            if isinstance(slice, ms.Tensor):
                slice = slice.asnumpy()
            path = folder.joinpath(f"{filename}.{extension}")
            save_fn(slice, path)


def save_intermediates(output_queue: queue.Queue) -> None:
    while True:
        try:
            item = output_queue.get(timeout=30)
            if item is None:
                break
            serialize_artifacts(**item)

        except queue.Empty:
            continue


def main():
    args = get_args()
    set_seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    tmp_dir = output_dir.joinpath("tmp")

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Create task queue for non-blocking serializing of artifacts
    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=args.num_artifact_workers)
    save_future = save_thread.submit(save_intermediates, output_queue)

    # Initialize distributed processing
    init_distributed_device(args)

    # Create folders where intermediate tensors from each rank will be saved
    images_dir = tmp_dir.joinpath(f"images/{args.rank}")
    image_latents_dir = tmp_dir.joinpath(f"image_latents/{args.rank}")
    videos_dir = tmp_dir.joinpath(f"videos/{args.rank}")
    video_latents_dir = tmp_dir.joinpath(f"video_latents/{args.rank}")
    prompts_dir = tmp_dir.joinpath(f"prompts/{args.rank}")
    prompt_embeds_dir = tmp_dir.joinpath(f"prompt_embeds/{args.rank}")

    images_dir.mkdir(parents=True, exist_ok=True)
    image_latents_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_latents_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_embeds_dir.mkdir(parents=True, exist_ok=True)

    weight_dtype = DTYPE_MAPPING[args.dtype]
    target_fps = args.target_fps

    # 1. Prepare models
    if args.save_latents_and_embeddings:
        tokenizer = T5Tokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            args.model_id, subfolder="text_encoder", mindspore_dtype=weight_dtype
        )

        vae = AutoencoderKLCogVideoX.from_pretrained(args.model_id, subfolder="vae", mindspore_dtype=weight_dtype)

        if args.use_slicing:
            vae.enable_slicing()
        if args.use_tiling:
            vae.enable_tiling()

    # 2. Dataset
    dataset_init_kwargs = {
        "data_root": args.data_root,
        "dataset_file": args.dataset_file,
        "caption_column": args.caption_column,
        "video_column": args.video_column,
        "max_num_frames": args.max_num_frames,
        "id_token": args.id_token,
        "height_buckets": args.height_buckets,
        "width_buckets": args.width_buckets,
        "frame_buckets": args.frame_buckets,
        "load_tensors": False,
        "random_flip": args.random_flip,
        "image_to_video": args.save_image_latents,
        "tokenizer": tokenizer,
        "max_sequence_length": args.max_sequence_length,
        "use_rotary_positional_embeddings": False,
    }
    if args.video_reshape_mode is None:
        dataset = VideoDatasetWithResizing(**dataset_init_kwargs)
    else:
        dataset = VideoDatasetWithResizeAndRectangleCrop(
            video_reshape_mode=args.video_reshape_mode, **dataset_init_kwargs
        )

    original_dataset_size = len(dataset)

    # Split data among GPUs
    if args.world_size > 1:
        samples_per_gpu = original_dataset_size // args.world_size
        start_index = args.rank * samples_per_gpu
        end_index = start_index + samples_per_gpu
        if args.rank == args.world_size - 1:
            end_index = original_dataset_size  # Make sure the last GPU gets the remaining data

        # Slice the data
        dataset.prompts = dataset.prompts[start_index:end_index]
        dataset.video_paths = dataset.video_paths[start_index:end_index]
    else:
        pass

    rank_dataset_size = len(dataset)

    # 3. Dataloader
    def collate_fn(data):
        prompts = [x["prompt"] for x in data]
        prompts = np.stack(prompts)

        text_input_ids = [x["text_input_ids"] for x in data]
        text_input_ids = np.stack(text_input_ids)

        videos = [x["video"] for x in data]
        videos = np.stack(videos)

        if args.save_image_latents:
            images = [x["image"] for x in data]
            images = np.stack(images)
            return videos, prompts, text_input_ids, images
        else:
            return videos, prompts, text_input_ids

    dataloader = GeneratorDataset(
        dataset,
        column_names=["examples"],
        num_parallel_workers=args.dataloader_num_workers,
    ).batch(
        batch_size=1,
        per_batch_map=lambda examples, batch_info: collate_fn(examples),
        input_columns=["examples"],
        output_columns=["videos", "prompts", "text_input_ids", "images"]
        if args.save_image_latents
        else ["videos", "prompts", "text_input_ids"],
    )
    dataloader_iter = dataloader.create_tuple_iterator()

    # 4. Compute latents and embeddings and save
    if args.rank == 0:
        iterator = tqdm(
            dataloader_iter, desc="Encoding", total=(rank_dataset_size + args.batch_size - 1) // args.batch_size
        )
    else:
        iterator = dataloader_iter

    for step, batch in enumerate(iterator):
        try:
            images = None
            image_latents = None
            video_latents = None
            prompt_embeds = None

            if args.save_image_latents:
                images = batch[-1].to(weight_dtype)
                images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            videos = batch[0].to(weight_dtype)
            videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            prompts = batch[1].asnumpy().tolist()

            text_input_ids = batch[2]

            # Encode videos & images
            if args.save_latents_and_embeddings:
                if args.use_slicing:
                    if args.save_image_latents:
                        encoded_slices = [vae._encode(image_slice) for image_slice in images.split(1)]
                        image_latents = ops.cat(encoded_slices)
                        image_latents = image_latents.to(dtype=weight_dtype).float().asnumpy()

                    encoded_slices = [vae._encode(video_slice) for video_slice in videos.split(1)]
                    video_latents = ops.cat(encoded_slices)

                else:
                    if args.save_image_latents:
                        image_latents = vae._encode(images)
                        image_latents = image_latents.to(dtype=weight_dtype).float().asnumpy()

                    video_latents = vae._encode(videos)

                video_latents = video_latents.to(dtype=weight_dtype).float().asnumpy()

                # Encode prompts
                prompt_embeds = (
                    compute_prompt_embeddings(
                        text_encoder,
                        text_input_ids,
                        dtype=weight_dtype,
                    )
                    .float()
                    .asnumpy()
                )

            if images is not None:
                images = (images.permute(0, 2, 3, 4, 1) + 1) / 2 * 255  # [B, F, H, W, C]
                images = images.asnumpy().astype(np.uint8)

            videos = (videos.permute(0, 2, 3, 4, 1) + 1) / 2 * 255  # [B, F, H, W, C]
            videos = videos.asnumpy().astype(np.uint8)

            # All tensors have been converted to numpy ndarray to save
            output_queue.put(
                {
                    "batch_size": len(prompts),
                    "fps": target_fps,
                    "images_dir": images_dir,
                    "image_latents_dir": image_latents_dir,
                    "videos_dir": videos_dir,
                    "video_latents_dir": video_latents_dir,
                    "prompts_dir": prompts_dir,
                    "prompt_embeds_dir": prompt_embeds_dir,
                    "images": images,
                    "image_latents": image_latents,
                    "videos": videos,
                    "video_latents": video_latents,
                    "prompts": prompts,
                    "prompt_embeds": prompt_embeds,
                }
            )

        except Exception:
            print("-------------------------")
            print(f"An exception occurred while processing data: {args.rank=}, {args.world_size=}, {step=}")
            traceback.print_exc()
            print("-------------------------")

    # 5. Complete distributed processing
    if args.world_size > 1:
        ops.Barrier()()

    output_queue.put(None)
    save_thread.shutdown(wait=True)
    save_future.result()

    # 6. Combine results from each rank
    if is_master(args):
        print(
            f"Completed preprocessing latents and embeddings. Temporary files from all ranks saved to `{tmp_dir.as_posix()}`"
        )

        # Move files from each rank to common directory
        for subfolder, extension in [
            ("images", "png"),
            ("image_latents", "npy"),
            ("videos", "mp4"),
            ("video_latents", "npy"),
            ("prompts", "txt"),
            ("prompt_embeds", "npy"),
            ("videos", "txt"),
        ]:
            tmp_subfolder = tmp_dir.joinpath(subfolder)
            combined_subfolder = output_dir.joinpath(subfolder)
            combined_subfolder.mkdir(parents=True, exist_ok=True)
            pattern = f"*.{extension}"

            for file in tmp_subfolder.rglob(pattern):
                file.replace(combined_subfolder / file.name)

        # Remove temporary directories
        def rmdir_recursive(dir: pathlib.Path) -> None:
            for child in dir.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    rmdir_recursive(child)
            dir.rmdir()

        rmdir_recursive(tmp_dir)

        # Combine prompts and videos into individual text files and single jsonl
        prompts_folder = output_dir.joinpath("prompts")
        prompts = []
        stems = []

        for filename in prompts_folder.rglob("*.txt"):
            with open(filename, "r") as file:
                prompts.append(file.read().strip())
            stems.append(filename.stem)

        prompts_txt = output_dir.joinpath("prompts.txt")
        videos_txt = output_dir.joinpath("videos.txt")
        data_jsonl = output_dir.joinpath("data.jsonl")

        with open(prompts_txt, "w") as file:
            for prompt in prompts:
                file.write(f"{prompt}\n")

        with open(videos_txt, "w") as file:
            for stem in stems:
                file.write(f"videos/{stem}.mp4\n")

        with open(data_jsonl, "w") as file:
            for prompt, stem in zip(prompts, stems):
                video_metadata_txt = output_dir.joinpath(f"videos/{stem}.txt")
                with open(video_metadata_txt, "r", encoding="utf-8") as metadata_file:
                    metadata = json.loads(metadata_file.read())

                data = {
                    "prompt": prompt,
                    "prompt_embed": f"prompt_embeds/{stem}.npy",
                    "image": f"images/{stem}.png",
                    "image_latent": f"image_latents/{stem}.npy",
                    "video": f"videos/{stem}.mp4",
                    "video_latent": f"video_latents/{stem}.npy",
                    "metadata": metadata,
                }
                file.write(json.dumps(data) + "\n")

        print(f"Completed preprocessing. All files saved to `{output_dir.as_posix()}`")


if __name__ == "__main__":
    main()
