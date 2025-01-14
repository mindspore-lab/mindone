import csv
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

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
        elif isinstance(size, int):
            max_size_hw = None
        else:
            raise ValueError("Expect size to be int or tuple of (h, w)")
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


def align_to(value, alignment):
    """align hight, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.ceil(value / alignment) * alignment)


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
        try:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                raise FileNotFoundError(f"Expect that {folder_path} exist!")

            for file_path in folder_path.rglob("*"):
                if file_path.is_file() and not (file_path.name.startswith(prefix) or file_path.suffix == ".txt"):
                    folder.append(str(file_path))

            folder_with_basename = [(os.path.basename(path), path) for path in folder]
            folder_sorted = [path for _, path in sorted(folder_with_basename)]

            return folder_sorted

        except Exception as e:
            print(f"An error occurred: {e}")
            return []


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


def pad_to_multiple(number, ds_stride):
    remainder = number % ds_stride
    if remainder == 0:
        return number
    else:
        padding = ds_stride - remainder
        return number + padding


class Collate:
    def __init__(self, batch_size, args):
        self.batch_size = batch_size
        self.group_data = args.group_data
        self.force_resolution = args.force_resolution

        self.max_height = args.max_height
        self.max_width = args.max_width
        self.ae_stride = args.ae_stride

        self.ae_stride_t = args.ae_stride_t
        self.ae_stride_thw = (self.ae_stride_t, self.ae_stride, self.ae_stride)

        self.patch_size = args.patch_size
        self.patch_size_t = args.patch_size_t

        self.num_frames = args.num_frames
        self.max_thw = (self.num_frames, self.max_height, self.max_width)

    def package(self, batch):
        batch_tubes = [i["pixel_values"] for i in batch]  # b [c t h w]
        input_ids_1 = [i["input_ids_1"] for i in batch]  # b [1 l]
        cond_mask_1 = [i["cond_mask_1"] for i in batch]  # b [1 l]
        input_ids_2 = [i["input_ids_2"] for i in batch]  # b [1 l]
        cond_mask_2 = [i["cond_mask_2"] for i in batch]  # b [1 l]
        assert all([i is None for i in input_ids_2]) or all([i is not None for i in input_ids_2])
        assert all([i is None for i in cond_mask_2]) or all([i is not None for i in cond_mask_2])
        if all([i is None for i in input_ids_2]):
            input_ids_2 = None
        if all([i is None for i in cond_mask_2]):
            cond_mask_2 = None
        return batch_tubes, input_ids_1, cond_mask_1, input_ids_2, cond_mask_2

    def __call__(self, batch):
        batch_tubes, input_ids_1, cond_mask_1, input_ids_2, cond_mask_2 = self.package(batch)

        ds_stride = self.ae_stride * self.patch_size
        t_ds_stride = self.ae_stride_t * self.patch_size_t

        pad_batch_tubes, attention_mask, input_ids_1, cond_mask_1, input_ids_2, cond_mask_2 = self.process(
            batch_tubes,
            input_ids_1,
            cond_mask_1,
            input_ids_2,
            cond_mask_2,
            t_ds_stride,
            ds_stride,
            self.max_thw,
            self.ae_stride_thw,
        )
        assert not np.any(np.isnan(pad_batch_tubes)), "after pad_batch_tubes"
        if input_ids_2 is not None and cond_mask_2 is not None:
            return pad_batch_tubes, attention_mask, input_ids_1, cond_mask_1, input_ids_2, cond_mask_2
        else:
            return pad_batch_tubes, attention_mask, input_ids_1, cond_mask_1

    def process(
        self,
        batch_tubes,
        input_ids_1,
        cond_mask_1,
        input_ids_2,
        cond_mask_2,
        t_ds_stride,
        ds_stride,
        max_thw,
        ae_stride_thw,
    ):
        # pad to max multiple of ds_stride
        batch_input_size = [i.shape for i in batch_tubes]  # [(c t h w), (c t h w)]
        assert len(batch_input_size) == self.batch_size
        if self.group_data or self.batch_size == 1:  #
            len_each_batch = batch_input_size
            idx_length_dict = dict([*zip(list(range(self.batch_size)), len_each_batch)])
            count_dict = Counter(len_each_batch)
            if len(count_dict) != 1:
                sorted_by_value = sorted(count_dict.items(), key=lambda item: item[1])
                # import ipdb;ipdb.set_trace()
                # print(batch, idx_length_dict, count_dict, sorted_by_value)
                pick_length = sorted_by_value[-1][0]  # the highest frequency
                candidate_batch = [idx for idx, length in idx_length_dict.items() if length == pick_length]
                random_select_batch = [
                    random.choice(candidate_batch) for _ in range(len(len_each_batch) - len(candidate_batch))
                ]
                print(
                    batch_input_size,
                    idx_length_dict,
                    count_dict,
                    sorted_by_value,
                    pick_length,
                    candidate_batch,
                    random_select_batch,
                )
                pick_idx = candidate_batch + random_select_batch

                batch_tubes = [batch_tubes[i] for i in pick_idx]
                batch_input_size = [i.shape for i in batch_tubes]  # [(c t h w), (c t h w)]
                input_ids_1 = [input_ids_1[i] for i in pick_idx]  # b [1, l]
                cond_mask_1 = [cond_mask_1[i] for i in pick_idx]  # b [1, l]
                if input_ids_2 is not None:
                    input_ids_2 = [input_ids_2[i] for i in pick_idx]  # b [1, l]
                if cond_mask_2 is not None:
                    cond_mask_2 = [cond_mask_2[i] for i in pick_idx]  # b [1, l]

            for i in range(1, self.batch_size):
                assert batch_input_size[0] == batch_input_size[i]
            max_t = max([i[1] for i in batch_input_size])
            max_h = max([i[2] for i in batch_input_size])
            max_w = max([i[3] for i in batch_input_size])
        else:
            max_t, max_h, max_w = max_thw
        pad_max_t, pad_max_h, pad_max_w = (
            pad_to_multiple(max_t - 1 + self.ae_stride_t, t_ds_stride),
            pad_to_multiple(max_h, ds_stride),
            pad_to_multiple(max_w, ds_stride),
        )
        pad_max_t = pad_max_t + 1 - self.ae_stride_t
        each_pad_t_h_w = [[pad_max_t - i.shape[1], pad_max_h - i.shape[2], pad_max_w - i.shape[3]] for i in batch_tubes]
        pad_batch_tubes = [
            np.pad(im, [[0, 0]] * (len(im.shape) - 3) + [[0, pad_t], [0, pad_h], [0, pad_w]], constant_values=0)
            for (pad_t, pad_h, pad_w), im in zip(each_pad_t_h_w, batch_tubes)
        ]
        pad_batch_tubes = np.stack(pad_batch_tubes, axis=0)

        max_tube_size = [pad_max_t, pad_max_h, pad_max_w]
        max_latent_size = [
            ((max_tube_size[0] - 1) // ae_stride_thw[0] + 1),
            max_tube_size[1] // ae_stride_thw[1],
            max_tube_size[2] // ae_stride_thw[2],
        ]
        valid_latent_size = [
            [
                int(math.ceil((i[1] - 1) / ae_stride_thw[0])) + 1,
                int(math.ceil(i[2] / ae_stride_thw[1])),
                int(math.ceil(i[3] / ae_stride_thw[2])),
            ]
            for i in batch_input_size
        ]
        attention_mask = [
            np.pad(
                np.ones(i, dtype=pad_batch_tubes.dtype),
                [[0, 0]] * (len(i) - 3)
                + [[0, max_latent_size[0] - i[0]], [0, max_latent_size[1] - i[1]], [0, max_latent_size[2] - i[2]]],
                constant_values=0,
            )
            for i in valid_latent_size
        ]
        attention_mask = np.stack(attention_mask, axis=0)  # b t h w
        if self.batch_size == 1 or self.group_data:
            if not np.all(attention_mask.astype(np.bool_)):
                print(
                    batch_input_size,
                    (max_t, max_h, max_w),
                    (pad_max_t, pad_max_h, pad_max_w),
                    each_pad_t_h_w,
                    max_latent_size,
                    valid_latent_size,
                )
            assert np.all(attention_mask.astype(np.bool_))

        input_ids_1 = np.stack(input_ids_1)  # b 1 l
        cond_mask_1 = np.stack(cond_mask_1)  # b 1 l
        input_ids_2 = np.stack(input_ids_2) if input_ids_2 is not None else input_ids_2  # b 1 l
        cond_mask_2 = np.stack(cond_mask_2) if cond_mask_2 is not None else cond_mask_2  # b 1 l

        return pad_batch_tubes, attention_mask, input_ids_1, cond_mask_1, input_ids_2, cond_mask_2


def group_data_fun(lengths, generator=None):
    # counter is decrease order
    counter = Counter(lengths)  # counter {'1x256x256': 3, ''}   lengths ['1x256x256', '1x256x256', '1x256x256', ...]
    grouped_indices = defaultdict(list)
    for idx, item in enumerate(lengths):  # group idx to a list
        grouped_indices[item].append(idx)

    grouped_indices = dict(grouped_indices)  # {'1x256x256': [0, 1, 2], ...}
    sorted_indices = [grouped_indices[item] for (item, _) in sorted(counter.items(), key=lambda x: x[1], reverse=True)]

    # shuffle in each group
    shuffle_sorted_indices = []
    for indice in sorted_indices:
        shuffle_idx = generator.permutation(len(indice)).tolist()
        shuffle_sorted_indices.extend([indice[idx] for idx in shuffle_idx])
    return shuffle_sorted_indices


def last_group_data_fun(shuffled_megabatches, lengths):
    # lengths ['1x256x256', '1x256x256', '1x256x256' ...]
    re_shuffled_megabatches = []
    # print('shuffled_megabatches', len(shuffled_megabatches))
    for i_megabatch, megabatch in enumerate(shuffled_megabatches):
        re_megabatch = []
        for i_batch, batch in enumerate(megabatch):
            assert len(batch) != 0

            len_each_batch = [lengths[i] for i in batch]  # ['1x256x256', '1x256x256']
            idx_length_dict = dict([*zip(batch, len_each_batch)])  # {0: '1x256x256', 100: '1x256x256'}
            count_dict = Counter(len_each_batch)  # {'1x256x256': 2} or {'1x256x256': 1, '1x768x256': 1}
            if len(count_dict) != 1:
                sorted_by_value = sorted(
                    count_dict.items(), key=lambda item: item[1]
                )  # {'1x256x256': 1, '1x768x256': 1}
                # import ipdb;ipdb.set_trace()
                # print(batch, idx_length_dict, count_dict, sorted_by_value)
                pick_length = sorted_by_value[-1][0]  # the highest frequency
                candidate_batch = [idx for idx, length in idx_length_dict.items() if length == pick_length]
                random_select_batch = [
                    random.choice(candidate_batch) for i in range(len(len_each_batch) - len(candidate_batch))
                ]
                # print(batch, idx_length_dict, count_dict, sorted_by_value, pick_length, candidate_batch, random_select_batch)
                batch = candidate_batch + random_select_batch
                # print(batch)

            for i in range(1, len(batch) - 1):
                # if not lengths[batch[0]] == lengths[batch[i]]:
                #     print(batch, [lengths[i] for i in batch])
                #     import ipdb;ipdb.set_trace()
                assert lengths[batch[0]] == lengths[batch[i]]
            re_megabatch.append(batch)
        re_shuffled_megabatches.append(re_megabatch)

    # for megabatch, re_megabatch in zip(shuffled_megabatches, re_shuffled_megabatches):
    #     for batch, re_batch in zip(megabatch, re_megabatch):
    #         for i, re_i in zip(batch, re_batch):
    #             if i != re_i:
    #                 print(i, re_i)
    return re_shuffled_megabatches


def split_to_even_chunks(megabatch, lengths, world_size, batch_size):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    # batch_size=2, world_size=2
    # [1, 2, 3, 4] -> [[1, 2], [3, 4]]
    # [1, 2, 3] -> [[1, 2], [3]]
    # [1, 2] -> [[1], [2]]
    # [1] -> [[1], []]
    chunks = [megabatch[i::world_size] for i in range(world_size)]

    pad_chunks = []
    for idx, chunk in enumerate(chunks):
        if batch_size != len(chunk):
            assert batch_size > len(chunk)
            if len(chunk) != 0:  # [[1, 2], [3]] -> [[1, 2], [3, 3]]
                chunk = chunk + [random.choice(chunk) for _ in range(batch_size - len(chunk))]
            else:
                chunk = random.choice(pad_chunks)  # [[1], []] -> [[1], [1]]
                print(chunks[idx], "->", chunk)
        pad_chunks.append(chunk)
    return pad_chunks


def get_length_grouped_indices(
    lengths,
    batch_size,
    world_size,
    gradient_accumulation_size,
    initial_global_step,
    generator=None,
    group_data=False,
    seed=42,
):
    if generator is None:
        generator = np.random.default_rng(seed)  # every rank will generate a fixed order but random index
    # print('lengths', lengths)

    if group_data:
        indices = group_data_fun(lengths, generator)
    else:
        indices = generator.permutation(len(lengths)).tolist()
    # print('indices', len(indices))

    # print('sort indices', len(indices))
    # print('sort indices', indices)
    # print('sort lengths', [lengths[i] for i in indices])

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    # import ipdb;ipdb.set_trace()
    # print('megabatches', len(megabatches))
    # print('\nmegabatches', megabatches)
    # megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    # import ipdb;ipdb.set_trace()
    # print('sort megabatches', len(megabatches))
    # megabatches_len = [[lengths[i] for i in megabatch] for megabatch in megabatches]
    # print(f'\nrank {accelerator.process_index} sorted megabatches_len', megabatches_len[0], megabatches_len[1], megabatches_len[-2], megabatches_len[-1])
    # import ipdb;ipdb.set_trace()
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size, batch_size) for megabatch in megabatches]
    # import ipdb;ipdb.set_trace()
    # print('nsplit_to_even_chunks megabatches', len(megabatches))
    # print('\nsplit_to_even_chunks megabatches', megabatches)
    # split_to_even_chunks_len = [[[lengths[i] for i in batch] for batch in megabatch] for megabatch in megabatches]
    # print(f'\nrank {accelerator.process_index} split_to_even_chunks_len', split_to_even_chunks_len[0],
    #  split_to_even_chunks_len[1], split_to_even_chunks_len[-2], split_to_even_chunks_len[-1])
    # print('\nsplit_to_even_chunks len', split_to_even_chunks_len)
    # return [i for megabatch in megabatches for batch in megabatch for i in batch]

    indices_mega = generator.permutation(len(megabatches)).tolist()
    # print(f'rank {accelerator.process_index} seed {seed}, len(megabatches) {len(megabatches)}, indices_mega, {indices_mega[:50]}')
    shuffled_megabatches = [megabatches[i] for i in indices_mega]
    # shuffled_megabatches_len = [
    #    [[lengths[i] for i in batch] for batch in megabatch] for megabatch in shuffled_megabatches
    # ]
    # print(f'\nrank {accelerator.process_index} sorted shuffled_megabatches_len', shuffled_megabatches_len[0],
    # shuffled_megabatches_len[1], shuffled_megabatches_len[-2], shuffled_megabatches_len[-1])

    # import ipdb;ipdb.set_trace()
    # print('shuffled_megabatches', len(shuffled_megabatches))
    if group_data:
        shuffled_megabatches = last_group_data_fun(shuffled_megabatches, lengths)
        # group_shuffled_megabatches_len = [
        #    [[lengths[i] for i in batch] for batch in megabatch] for megabatch in shuffled_megabatches
        # ]
        # print(f'\nrank {accelerator.process_index} group_shuffled_megabatches_len', group_shuffled_megabatches_len[0],
        # group_shuffled_megabatches_len[1], group_shuffled_megabatches_len[-2], group_shuffled_megabatches_len[-1])

    # import ipdb;ipdb.set_trace()
    initial_global_step = initial_global_step * gradient_accumulation_size
    # print('shuffled_megabatches', len(shuffled_megabatches))
    # print('have been trained idx:', len(shuffled_megabatches[:initial_global_step]))
    # print('shuffled_megabatches[:10]', shuffled_megabatches[:10])
    # print('have been trained idx:', shuffled_megabatches[:initial_global_step])
    shuffled_megabatches = shuffled_megabatches[initial_global_step:]
    print(f"Skip the data of {initial_global_step} step!")
    # print('after shuffled_megabatches', len(shuffled_megabatches))
    # print('after shuffled_megabatches[:10]', shuffled_megabatches[:10])

    # print('\nshuffled_megabatches', shuffled_megabatches)
    # import ipdb;ipdb.set_trace()
    # print('\nshuffled_megabatches len', [[i, lengths[i]] for megabatch in shuffled_megabatches for batch in megabatch for i in batch])
    return [
        i for megabatch in shuffled_megabatches for batch in megabatch for i in batch
    ]  # return epoch indices in a single list


class LengthGroupedSampler:
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        gradient_accumulation_size: int,
        initial_global_step: int,
        lengths: Optional[List[int]] = None,
        group_data=False,
        generator=None,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.initial_global_step = initial_global_step
        self.gradient_accumulation_size = gradient_accumulation_size
        self.lengths = lengths
        self.group_data = group_data
        self.generator = generator
        # print('self.lengths, self.initial_global_step, self.batch_size, self.world_size, self.gradient_accumulation_size',
        #       len(self.lengths), self.initial_global_step, self.batch_size, self.world_size, self.gradient_accumulation_size)

    def __len__(self):
        return (
            len(self.lengths)
            - self.initial_global_step * self.batch_size * self.world_size * self.gradient_accumulation_size
        )

    def __iter__(self):
        indices = get_length_grouped_indices(
            self.lengths,
            self.batch_size,
            self.world_size,
            self.gradient_accumulation_size,
            self.initial_global_step,
            group_data=self.group_data,
            generator=self.generator,
        )

        return iter(indices)
