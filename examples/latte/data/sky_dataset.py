import logging
import os
import random

import mindspore as ms

logger = logging.getLogger()
import numpy as np
from PIL import Image

from .dataset import create_video_transforms

IMG_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class SkyDataset:
    """A dataset for Sky Timelapse
    data_path: (str) a folder path where the videos frames are all stored.
    sample_size: (int, default=256) image size
    sample_stride: (int, default=4) sample stride, should be positive
    sample_n_frames: (int, default=16) the number of sampled frames, only applies when `frame_index_sampler` is None.
    transform_backend: (str, default="al") one of transformation backends in [ms, pt, al]. "al" is recommended.
    use_safer_augment: (bool, default=False), whether to use safe augmentation. If True, it will disable random horizontal flip.
    image_video_joint: (bool, default=False), whether to use image-video-joint training. If True, the dataset will return the concatenation of `video_frames`
        and randomly-sampled `images` as the pixel values (concatenated at the frame axis).
    use_image_num: (int, default=None), the number of randomly-sampled images in image-video-joint training.
    """

    def __init__(
        self,
        data_path,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
        transform_backend="al",  # ms, pt, al
        use_safer_augment=False,
        image_video_joint=False,
        use_image_num=None,
    ):
        logger.info(f"loading frames from {data_path} ...")
        self.data_path = data_path
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.transform_backend = transform_backend
        # it should match the transformation used in SD/VAE pretraining, especially for normalization
        self.pixel_transforms = create_video_transforms(
            sample_size[0],
            sample_size[1],
            sample_n_frames,
            interpolation="bicubic",
            backend=transform_backend,
            use_safer_augment=use_safer_augment,
            apply_same_transform=True,
        )

        self.image_video_joint = image_video_joint
        if self.image_video_joint:
            self.data_all, self.video_frame_all = self.load_video_frames(self.data_path)
            random.shuffle(self.video_frame_all)
            self.use_image_num = use_image_num
            assert (
                isinstance(self.use_image_num, int) and self.use_image_num > 0
            ), "Expect to get use_image_num as a positive integer"
        else:
            self.data_all, _ = self.load_video_frames(self.data_path)

        logger.info(f"{self.video_num} videos ({self.video_frame_num} frames) are loaded.")

    def load_video_frames(self, dataroot):
        data_all = []
        frames_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            frames = meta[2]
            frames = [os.path.join(root, item) for item in frames if is_image_file(item)]
            try:
                frames = sorted(frames, key=lambda item: int(os.path.basename(item).split(".")[0].split("_")[-1]))
            except Exception:
                print(meta[0])  # root
                print(meta[2])  # files
            if len(frames) > max(0, self.sample_n_frames * self.sample_stride):
                data_all.append(frames)
                for frame in frames:
                    frames_all.append(frame)
        self.video_num = len(data_all)
        self.video_frame_num = len(frames_all)
        return data_all, frames_all

    def __len__(self):
        if self.image_video_joint:
            return self.video_frame_num
        else:
            return self.video_num

    def __getitem__(self, index):
        if self.image_video_joint:
            video_index = index % self.video_num
        else:
            video_index = index
        vframes = self.data_all[video_index]
        video_length = len(vframes)

        # Sampling video frames
        clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        frame_indice = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        select_video_frames = [vframes[ind] for ind in frame_indice]

        video_frames = []
        for path in select_video_frames:
            img = Image.open(path).convert("RGB")
            video_frame = np.array(img, dtype=np.uint8, copy=True)
            video_frames.append(video_frame)
        pixel_values = np.stack(video_frames, axis=0)  # (f, h, w, c)
        pixel_values = self.apply_transform(pixel_values)

        # get random frames if needed
        if self.image_video_joint:
            images = []
            for i in range(self.use_image_num):
                while True:
                    try:
                        video_frame_path = self.video_frame_all[index + i]
                        image = np.array(Image.open(video_frame_path), dtype=np.uint8, copy=True)
                        images.append(image)
                        break
                    except Exception:
                        index = random.randint(0, self.video_frame_num - self.use_image_num)
            images_values = np.stack(images, axis=0)  # (n, h, w, c)
            images_values = self.apply_transform(images_values)
            pixel_values = np.concatenate([pixel_values, images_values], axis=0)  # (f+n, h, w, c)

        # normallization
        pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)
        return pixel_values

    def apply_transform(self, pixel_values):
        transform_func = self.pixel_transforms
        if self.transform_backend == "pt":
            import torch

            pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
            pixel_values = transform_func(pixel_values)
            pixel_values = pixel_values.numpy()
        elif self.transform_backend == "al":
            # NOTE:it's to ensure augment all frames in a video in the same way.
            # ref: https://albumentations.ai/docs/examples/example_multi_target/

            inputs = {"image": pixel_values[0]}
            num_frames = len(pixel_values)
            for i in range(num_frames - 1):
                inputs[f"image{i}"] = pixel_values[i + 1]

            output = transform_func(**inputs)

            pixel_values = np.stack(list(output.values()), axis=0)
            # (f h w c) -> (f c h w)
            pixel_values = np.transpose(pixel_values, (0, 3, 1, 2))
        else:
            raise NotImplementedError
        return pixel_values


def create_dataloader(config, device_num=1, rank_id=0, **kwargs):
    dataset = SkyDataset(
        config["data_folder"],
        sample_size=config["sample_size"],
        sample_stride=config["sample_stride"],
        sample_n_frames=config["sample_n_frames"],
        use_safer_augment=config["use_safer_augment"],
        image_video_joint=config["image_video_joint"],
        use_image_num=config.get("use_image_num", None),
    )
    data_name = "video"
    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=[
            data_name,
        ],
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=config["shuffle"],
        num_parallel_workers=config["num_parallel_workers"],
        max_rowsize=config["max_rowsize"],  # video data require larger rowsize
    )

    dl = dataloader.batch(
        config["batch_size"],
        drop_remainder=True,
    )

    return dl
