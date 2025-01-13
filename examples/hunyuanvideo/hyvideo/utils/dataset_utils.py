import csv
import os
import os.path as osp

import decord
import numpy as np
from decord import VideoReader

import mindspore as ms


def create_video_transforms(
    size, crop_size, num_frames, interpolation="bicubic", backend="al", disable_flip=True, random_crop=False
):
    """
    pipeline: flip -> resize -> crop
    NOTE: we change interpolation to bicubic for its better precision and used in SD. TODO: check impact on performance
    Args:
        size: resize to this size
        crop_size: tuple or integer, crop to this size.
        num_frames: number of frames in the video.
        interpolation: interpolation method.
        backend: backend to use. Currently only support albumentations.
        disable_flip: disable flip.
        random_crop: crop randomly. If False, crop center.
    """
    if isinstance(crop_size, (tuple, list)):
        h, w = crop_size
    else:
        h, w = crop_size, crop_size

    if backend == "al":
        # expect rgb image in range 0-255, shape (h w c)
        import albumentations
        import cv2
        from albumentations import CenterCrop, HorizontalFlip, RandomCrop, SmallestMaxSize

        targets = {"image{}".format(i): "image" for i in range(num_frames)}
        mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
        if isinstance(size, (tuple, list)):
            assert len(size) == 2, "Expect size should be a tuple or integer of (h, w)"
            max_size_hw = size
            size = None
        transforms_list = [
            SmallestMaxSize(max_size=size, max_size_hw=max_size_hw, interpolation=mapping[interpolation]),
            CenterCrop(h, w) if not random_crop else RandomCrop(h, w),
        ]
        if not disable_flip:
            transforms_list.insert(0, HorizontalFlip(p=0.5))
        pixel_transforms = albumentations.Compose(
            transforms_list,
            additional_targets=targets,
        )
    else:
        raise NotImplementedError

    return pixel_transforms


def create_image_transforms(
    size, crop_size, interpolation="bicubic", backend="al", random_crop=False, disable_flip=True
):
    if isinstance(crop_size, (tuple, list)):
        h, w = crop_size
    else:
        h, w = crop_size, crop_size

    if backend == "pt":
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        mapping = {"bilinear": InterpolationMode.BILINEAR, "bicubic": InterpolationMode.BICUBIC}

        pixel_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=mapping[interpolation]),
                transforms.CenterCrop((h, w)) if not random_crop else transforms.RandomCrop((h, w)),
            ]
        )
    else:
        # expect rgb image in range 0-255, shape (h w c)
        import albumentations
        import cv2
        from albumentations import CenterCrop, HorizontalFlip, RandomCrop, SmallestMaxSize

        mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
        transforms_list = [
            SmallestMaxSize(max_size=size, interpolation=mapping[interpolation]),
            CenterCrop(crop_size, crop_size) if not random_crop else RandomCrop(crop_size, crop_size),
        ]
        if not disable_flip:
            transforms_list.insert(0, HorizontalFlip(p=0.5))

        pixel_transforms = albumentations.Compose(transforms)

    return pixel_transforms


IMG_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename, ctx=self.ctx, num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"sr={self.sr}," f"num_threads={self.num_threads})"
        return repr_str


class VideoPairDataset:
    """
    A Video dataset that reads from both the real and generated video folders, and return a video pair
    """

    def __init__(
        self,
        real_video_dir,
        generated_video_dir,
        num_frames,
        real_data_file_path=None,
        sample_rate=1,
        crop_size=None,
        resolution=128,
        output_columns=["real", "generated"],
    ) -> None:
        super().__init__()
        if real_data_file_path is not None:
            print(f"Loading videos from data file {real_data_file_path}")
            self.parse_data_file(real_data_file_path)
            self.read_from_data_file = True
        else:
            self.real_video_files = self.combine_without_prefix(real_video_dir)
            self.read_from_data_file = False
        self.generated_video_files = self.combine_without_prefix(generated_video_dir)
        assert (
            len(self.real_video_files) == len(self.generated_video_files) and len(self.real_video_files) > 0
        ), "Expect that the real and generated folders are not empty and contain the equal number of videos!"
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.crop_size = crop_size
        self.short_size = resolution
        self.output_columns = output_columns
        self.real_video_dir = real_video_dir

        self.pixel_transforms = create_video_transforms(
            size=self.short_size,
            crop_size=crop_size,
            random_crop=False,
            disable_flip=True,
            num_frames=num_frames,
            backend="al",
        )

    def __len__(self):
        return len(self.real_video_files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        if self.read_from_data_file:
            video_dict = self.real_video_files[index]
            video_fn = video_dict["video"]
            real_video_file = os.path.join(self.real_video_dir, video_fn)
        else:
            real_video_file = self.real_video_files[index]
        generated_video_file = self.generated_video_files[index]
        if os.path.basename(real_video_file).split(".")[0] != os.path.basename(generated_video_file).split(".")[0]:
            print(
                f"Warning! video file name mismatch! real and generated {os.path.basename(real_video_file)} and {os.path.basename(generated_video_file)}"
            )
        real_video_tensor = self._load_video(real_video_file)
        generated_video_tensor = self._load_video(generated_video_file)
        return real_video_tensor.astype(np.float32), generated_video_tensor.astype(np.float32)

    def parse_data_file(self, data_file_path):
        if data_file_path.endswith(".csv"):
            with open(data_file_path, "r") as csvfile:
                self.real_video_files = list(csv.DictReader(csvfile))
        else:
            raise ValueError("Only support csv file now!")
        self.real_video_files = sorted(self.real_video_files, key=lambda x: os.path.basename(x["video"]))

    def _load_video(self, video_path):
        num_frames = self.num_frames
        sample_rate = self.sample_rate
        decord_vr = VideoReader(
            video_path,
        )
        total_frames = len(decord_vr)
        sample_frames_len = sample_rate * num_frames

        if total_frames >= sample_frames_len:
            s = 0
            e = s + sample_frames_len
            num_frames = num_frames
        else:
            s = 0
            e = total_frames
            num_frames = int(total_frames / sample_frames_len * num_frames)
            print(f"Video total number of frames {total_frames} is less than the target num_frames {sample_frames_len}")
            print(video_path)

        frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
        pixel_values = decord_vr.get_batch(frame_id_list).asnumpy()
        # video_data = video_data.transpose(0, 3, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        # NOTE:it's to ensure augment all frames in a video in the same way.
        # ref: https://albumentations.ai/docs/examples/example_multi_target/

        inputs = {"image": pixel_values[0]}
        for i in range(num_frames - 1):
            inputs[f"image{i}"] = pixel_values[i + 1]

        output = self.pixel_transforms(**inputs)

        pixel_values = np.stack(list(output.values()), axis=0)
        # (t h w c) -> (t c h w)
        pixel_values = np.transpose(pixel_values, (0, 3, 1, 2))
        pixel_values = pixel_values / 255.0
        return pixel_values

    def combine_without_prefix(self, folder_path, prefix="."):
        folder = []
        assert os.path.exists(folder_path), f"Expect that {folder_path} exist!"
        for name in os.listdir(folder_path):
            if name[0] == prefix or name.split(".")[1] == "txt":
                continue
            if osp.isfile(osp.join(folder_path, name)):
                folder.append(osp.join(folder_path, name))
        folder = sorted(folder, key=lambda x: os.path.basename(x))
        return folder


def create_dataloader(
    dataset,
    batch_size,
    ds_name="video",
    num_parallel_workers=12,
    max_rowsize=32,
    shuffle=True,
    device_num=1,
    rank_id=0,
    drop_remainder=True,
):
    """
    Args:
        ds_config, dataset config, args for ImageDataset or VideoDataset
        ds_name: dataset name, image or video
    """
    column_names = getattr(dataset, "output_columns", ["video"])
    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=column_names,
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=max_rowsize,
    )

    dl = dataloader.batch(
        batch_size,
        drop_remainder=drop_remainder,
    )

    return dl
