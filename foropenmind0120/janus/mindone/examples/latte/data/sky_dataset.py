import glob
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
        self.load_video_frames(self.data_path)
        if self.image_video_joint:
            random.shuffle(self.video_frame_all)
            self.use_image_num = use_image_num
            assert (
                isinstance(self.use_image_num, int) and self.use_image_num > 0
            ), f"Expect to get use_image_num as a positive integer, but got {self.use_image_num}"

        logger.info(f"{self.video_num} videos are loaded.")

    def load_video_frames(self, dataroot):
        data_all = []
        frames_all = []
        video_names = []
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
                # get video  name
                video_name = os.path.abspath(root).split(os.path.abspath(self.data_path))[-1]
                video_name = video_name.split("/")
                if len(video_name) > 1:
                    video_name = video_name[-1]
                assert len(video_name) > 0, "found empty video name!"
                video_names.append(video_name)

        self.video_num = len(data_all)
        self.video_frame_num = len(frames_all)
        self.video_names = video_names
        assert (
            len(self.video_names) == self.video_num and len(set(self.video_names)) == self.video_num
        ), f"video names should be {self.video_num} non-repetitive names"
        self.data_all = data_all
        self.video_frame_all = frames_all

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

    def traverse_single_video_frames(self, video_index):
        vframes = self.data_all[video_index]
        video_length = len(vframes)
        video_name = self.video_names[video_index]

        # Sampling video frames
        clips_indices = []
        start_idx = 0
        while start_idx + self.sample_n_frames < video_length:
            clips_indices.append([start_idx, start_idx + self.sample_n_frames])
            start_idx += self.sample_n_frames
        if start_idx < video_length:
            clips_indices.append([start_idx, video_length])
        assert len(clips_indices) > 0 and clips_indices[-1][-1] == video_length, "incorrect sampled clips!"

        for clip_indices in clips_indices:
            i, j = clip_indices
            frame_indice = list(range(i, j, 1))
            select_video_frames = [vframes[ind] for ind in frame_indice]
            video_frames = []
            for path in select_video_frames:
                img = Image.open(path).convert("RGB")
                video_frame = np.array(img, dtype=np.uint8, copy=True)
                video_frames.append(video_frame)
            pixel_values = np.stack(video_frames, axis=0)  # (f, h, w, c)
            pixel_values = self.apply_transform(pixel_values)
            pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)

            yield video_name, select_video_frames, {"video": pixel_values}


class SkyDatasetWithEmbeddingNumpy(SkyDataset):
    def load_video_frames(self, dataroot):
        self.video_dict = {}
        self.video_names = []
        self.video_frame_all = []
        num_filtered_videos = 0

        # load npz files first
        npz_files = glob.glob(os.path.join(dataroot, "*.npz"))
        if len(npz_files) > 0:
            for fp in npz_files:
                video_name = os.path.basename(fp).split(".")[0]
                self.video_dict[video_name] = {"npz": fp, "npy": []}

        # then load npy files
        if len(self.video_dict) > 0:
            # load from video_names folders
            video_folders = [os.path.join(dataroot, video_name) for video_name in self.video_dict]
        else:
            # load from all sub-folders
            # get video folders
            video_folders = []
            for file in os.listdir(dataroot):
                folder = os.path.join(dataroot, file)
                if os.path.isdir(folder) and len(os.listdir(folder)) > 0:
                    video_folders.append(folder)
                    assert len(file) > 0, "found empty video name!"
                    self.video_dict[video_name] = {"npz": None, "npy": []}

        for video_folder in video_folders:
            video_name = os.path.basename(video_folder)
            if os.path.exists(video_folder):
                frames = glob.glob(os.path.join(video_folder, "*.npy"))
                frames = sorted(frames, key=lambda item: int(os.path.basename(item).split(".")[0].split("_")[-1]))

                if len(frames) > max(0, self.sample_n_frames * self.sample_stride):
                    self.video_dict[video_name]["npy"] = frames
                    self.video_frame_all.extend(frames)
                else:
                    # filter videos that are too short
                    if video_name in self.video_dict:
                        del self.video_dict[video_name]
                    num_filtered_videos += 1

        self.video_num = len(self.video_dict)
        self.video_frame_num = len(self.video_frame_all)
        if self.video_frame_num == 0:
            # no npy file existent
            assert not self.image_video_joint, "Cannot apply image-video-joint training, because no frame num!"
        self.video_names = list(self.video_dict.keys())
        if num_filtered_videos:
            logger.info(
                f"{num_filtered_videos} videos were filtered out because the number of frames are smaller"
                f" than n_frames * sample_stride: {self.sample_n_frames * self.sample_stride}!"
            )

    def __getitem__(self, index):
        if self.image_video_joint:
            video_index = index % self.video_num
        else:
            video_index = index

        # get npz file if needed
        video_name = self.video_names[video_index]
        if self.video_dict[video_name]["npz"]:
            emb_fp = self.video_dict[video_name]["npz"]
            emb_data = np.load(emb_fp)
            if "video_latent" in emb_data:
                video_latent = emb_data["video_latent"]
                video_length = len(video_latent)

        if self.video_dict[video_name]["npy"]:
            emb_data = self.video_dict[video_name]["npy"]
            video_length = len(emb_data)
            video_latent = []

        # Sampling video frames
        clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        frame_indice = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        if len(video_latent):
            video_emb_train = video_latent[frame_indice]
        else:
            # load from npy files
            video_frames_paths = self.video_dict[video_name]["npy"]
            frames = [video_frames_paths[index] for index in frame_indice]
            for frame_path in frames:
                latent = np.load(frame_path)
                video_latent.append(latent)
            video_emb_train = np.stack(video_latent, axis=0)

        # get random frames if needed
        if self.image_video_joint:
            images_embeddings = []
            for i in range(self.use_image_num):
                while True:
                    try:
                        video_frame_path = self.video_frame_all[index + i]
                        image_emb = np.load(video_frame_path)
                        images_embeddings.append(image_emb)
                        break
                    except Exception:
                        index = random.randint(0, self.video_frame_num - self.use_image_num)
            images_embeddings = np.stack(images_embeddings, axis=0)
            video_emb_train = np.concatenate([video_emb_train, images_embeddings], axis=0)
        return video_emb_train.astype(np.float32)


class SelectFrameMap:
    def __init__(self, is_image=False, sample_n_frames=16, sample_stride=4):
        self.is_image = is_image
        self.sample_n_frames = sample_n_frames
        self.sample_stride = sample_stride

    def __call__(self, video_latent):
        video_length = len(video_latent)
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        video_emb_train = video_latent[batch_index]
        return video_emb_train


def create_dataloader(config, device_num=1, rank_id=0, return_dataset=False, **kwargs):
    if "train_data_type" not in config:
        # by default a video file dataset
        config["train_data_type"] = "video_file"

    if config["train_data_type"] == "mindrecord":
        data_files = []
        for file in os.listdir(config["data_folder"]):
            file_path = os.path.join(config["data_folder"], file)
            if os.path.isfile(file_path) and file.split(".")[-1] == "mindrecord":
                data_files.append(file_path)

        dataset = ms.dataset.MindDataset(
            dataset_files=data_files,
            columns_list=["video_latent"],
            num_shards=device_num,
            shard_id=rank_id,
            shuffle=config["shuffle"],
            num_parallel_workers=config["num_parallel_workers"],
        )
        select_frames_map = SelectFrameMap(
            is_image=False,
            sample_n_frames=config["sample_n_frames"],
            sample_stride=config["sample_stride"],
        )
        dataloader = dataset.map(operations=select_frames_map, input_columns=["video_latent"])

    else:
        if config["train_data_type"] == "video_file":
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
        elif config["train_data_type"] == "numpy":
            dataset = SkyDatasetWithEmbeddingNumpy(
                config["data_folder"],
                sample_size=config["sample_size"],
                sample_stride=config["sample_stride"],
                sample_n_frames=config["sample_n_frames"],
                use_safer_augment=config["use_safer_augment"],
                image_video_joint=config["image_video_joint"],
                use_image_num=config.get("use_image_num", None),
            )
            data_name = "video_latent"
        else:
            raise ValueError("Train data type {} is not supported!".format(config["train_data_type"]))

        dataloader = ms.dataset.GeneratorDataset(
            source=dataset,
            column_names=[data_name],
            num_shards=device_num,
            shard_id=rank_id,
            python_multiprocessing=True,
            shuffle=config["shuffle"],
            num_parallel_workers=config["num_parallel_workers"],
            max_rowsize=config["max_rowsize"],
        )

    dl = dataloader.batch(config["batch_size"], drop_remainder=True)

    if return_dataset:
        return dataset, dl
    else:
        return dl
