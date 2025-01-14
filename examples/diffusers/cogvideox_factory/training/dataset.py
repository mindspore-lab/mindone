from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer

from mindspore.dataset import transforms, vision
from mindspore.dataset.vision import Inter as InterpolationMode

from mindone.diffusers.utils import get_logger

from utils import pad_last_frame, prepare_rotary_positional_embeddings  # isort:skip

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("native")

logger = get_logger(__name__)

# Dynamic Shape is not supported right now.
# HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
# WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
# FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]

# Default Resolution same as THUDM--CogVideo.sat.configs.sft.data.params.video_size/fps
HEIGHT_BUCKETS = [480]
WIDTH_BUCKETS = [720]
FRAME_BUCKETS = [49]


class VideoDataset(object):
    def __init__(
        # Basic Arguments
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_sequence_length: Optional[int] = None,
        use_rotary_positional_embeddings: bool = False,
        vae_scale_factor_spatial: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        attention_head_dim: Optional[int] = None,
        base_height: int = 480,
        base_width: int = 720,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        # Tokenizer to process prompts to tokens
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        # Attributes about rotary_positional_embeddings generation
        self.use_rotary_positional_embeddings = use_rotary_positional_embeddings
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.attention_head_dim = attention_head_dim
        self.base_height = base_height
        self.base_width = base_width

        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]
        # We prepare RoPE in dataset
        self.prepare_ropes()

        # Two methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."  # noqa: E501
            )

        self.video_transforms = transforms.Compose(
            [
                vision.RandomHorizontalFlip(random_flip) if random_flip else self.identity_transform,
                self.scale_transform,
                vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
            ]
        )

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        if self.load_tensors:
            image_latents, video_latents, prompt_embeds = self._preprocess_video(self.video_paths[index])

            # This is hardcoded for now.
            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.shape[1]
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.shape[2] * 8
            width = video_latents.shape[3] * 8

            return {
                "prompt": None,
                "text_input_ids": prompt_embeds,
                "image": image_latents,
                "video": video_latents,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
                "rotary_positional_embeddings": self.ropes,
            }
        else:
            image, video, _ = self._preprocess_video(self.video_paths[index])

            prompt = self.id_token + self.prompts[index]
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="np",
            )
            text_input_ids = text_inputs.input_ids

            return {
                "prompt": prompt,
                "text_input_ids": text_input_ids.squeeze(),
                "image": image,
                "video": video,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
                "rotary_positional_embeddings": self.ropes,
            }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."  # noqa: E501
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."  # noqa: E501
            )

        return prompts, video_paths

    def _preprocess_video(self, path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.

        If returning latent/embedding, returns a [F, C, H, W] latent, and the prompt embedding of shape [S, D].
        F, C, H and W are the frames, channels, height and width of the latent, and S, D are the sequence length
        and embedding dimension of prompt embeddings.
        """
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)

            indices = list(range(0, video_num_frames, max(1, video_num_frames // self.max_num_frames)))
            frames = video_reader.get_batch(indices).asnumpy()
            frames = frames[: self.max_num_frames].astype(np.float32)
            frames = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
            frames = np.stack([self.video_transforms(frame) for frame in frames], axis=0)

            image = frames[:1].copy() if self.image_to_video else None

            return image, frames, None

    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.npy"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.npy
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."  # noqa: E501
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."  # noqa: E501
            )

        images = np.load(image_latent_filepath) if self.image_to_video else None
        latents = np.load(video_latent_filepath)
        embeds = np.load(embeds_filepath)

        return images, latents, embeds

    def prepare_ropes(self):
        if len(self.resolutions) != 1:
            raise NotImplementedError("Only support fixed frame and resolution now")

        f, h, w = self.resolutions[0]
        num_frames = (f - f % 2) / 4 + f % 2

        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=h,
                width=w,
                num_frames=int(num_frames),
                vae_scale_factor_spatial=self.vae_scale_factor_spatial,
                patch_size=self.patch_size,
                patch_size_t=self.patch_size_t,
                attention_head_dim=self.attention_head_dim,
                base_height=self.base_height,
                base_width=self.base_width,
            )
            if self.use_rotary_positional_embeddings
            else None
        )

        self.ropes = image_rotary_emb


class VideoDatasetWithResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _preprocess_video(self, path: Path) -> np.ndarray:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, max(1, video_num_frames // nearest_frame_bucket)))

            frames = video_reader.get_batch(frame_indices).asnumpy()
            frames = pad_last_frame(frames, nearest_frame_bucket).astype(np.float32)

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = np.stack([vision.Resize(size=nearest_res)(frame) for frame in frames], axis=0)
            frames_resized = np.ascontiguousarray(frames_resized.transpose(0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
            frames = np.stack([self.video_transforms(frame) for frame in frames_resized], axis=0)

            image = frames[:1].copy() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class VideoDatasetWithResizeAndRectangleCrop(VideoDataset):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = vision.Resize(
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )(arr)
        else:
            arr = vision.Resize(
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )(arr)

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = vision.Crop(coordinates=(top, left), size=(image_size[0], image_size[1]))(arr)
        return arr

    def _preprocess_video(self, path: Path) -> np.ndarray:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, max(1, video_num_frames // nearest_frame_bucket)))

            frames = video_reader.get_batch(frame_indices).asnumpy()
            frames = pad_last_frame(frames, nearest_frame_bucket).astype(np.float32)

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
            frames_resized = np.ascontiguousarray(frames_resized.transpose(0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
            frames = np.stack([self.video_transforms(frame) for frame in frames_resized], axis=0)

            image = frames[:1].copy() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]
