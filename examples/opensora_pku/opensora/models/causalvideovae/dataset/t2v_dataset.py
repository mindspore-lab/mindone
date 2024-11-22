import copy
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
from decord import VideoReader
from PIL import Image
from tqdm import tqdm

import mindspore as ms

from .transform import create_video_transforms, t5_text_preprocessing

logger = logging.getLogger(__name__)


class TextVideoDataset:
    def __init__(
        self,
        image_data,
        video_data,
        return_text_emb=False,
        return_vae_latent=False,
        sample_size=256,
        num_frames=16,
        use_image_num=0,
        use_img_from_vid=True,
        transform_backend="al",  # ms,  al
        tokenizer=None,
        disable_flip=True,
        model_max_length=300,
        use_text_preprocessing=True,
        filter_nonexistent=True,
    ):
        self.image_data = image_data
        self.video_data = video_data
        self.num_frames = num_frames
        self.transform_backend = transform_backend
        self.tokenizer = tokenizer

        self.return_text_emb = return_text_emb
        self.return_vae_latent = return_vae_latent
        assert not self.return_vae_latent, "Return vae latent cache is not supported!"
        self.filter_nonexistent = filter_nonexistent

        self.use_image_num = use_image_num
        self.use_img_from_vid = use_img_from_vid
        if self.num_frames != 1:
            self.vid_cap_list = self.get_vid_cap_list()
            if self.use_image_num != 0 and not self.use_img_from_vid:
                self.img_cap_list = self.get_img_cap_list()
        else:
            self.img_cap_list = self.get_img_cap_list()

        logger.info(f"Num data samples: {len(self)}")
        assert len(self) > 0, "Empty dataset is not supported!"
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        # it should match the transformation used in SD/VAE pretraining, especially for normalization
        self.pixel_transforms = create_video_transforms(
            sample_size[0],
            sample_size[1],
            self.num_frames + self.use_image_num,  # visual transform for both video and images
            interpolation="bicubic",
            backend=transform_backend,
            disable_flip=disable_flip,
        )
        # prepare replacement data
        max_attempts = 100
        self.prev_ok_sample = self.get_replace_data(max_attempts)
        self.require_update_prev = False
        self.model_max_length = model_max_length
        if use_text_preprocessing is None:
            self.text_preprocessing = lambda x: x.lower().strip()
        else:
            self.text_preprocessing = t5_text_preprocessing

    def get_replace_data(self, max_attempts=100):
        replace_data = None
        attempts = min(max_attempts, len(self))
        for idx in range(attempts):
            # TODO: uncomment after training verified
            try:
                pixel_values, text, mask = self.get_batch(idx)
                replace_data = copy.deepcopy((pixel_values, text, mask))
                break
            except Exception as e:
                print("\tError msg: {}".format(e), flush=True)

        assert replace_data is not None, f"Fail to preload sample in {attempts} attempts."

        return replace_data

    def parse_text_emb(self, npz):
        if not os.path.exists(npz):
            raise ValueError(
                f"text embedding file {npz} not found. Please check the text_emb_folder and make sure the text embeddings are already generated"
            )
        td = np.load(npz)
        text_emb = td["text_emb"]
        mask = td["mask"]
        if len(text_emb.shape) == 2:
            text_emb = text_emb[None, ...]
        if len(mask.shape) == 1:
            mask = mask[None, ...]

        return text_emb, mask  # (1, L, D), (1, L)

    def parse_video_latent(self, *args):
        raise NotImplementedError

    def video_read(self, video_path, frame_idx=None):
        # read video files (.mp4 or .gif) and sample frames randomly or given the frame-idx in the format (start_idx:end_idx)
        if not video_path.endswith(".mp4") or video_path.endswith(".gif"):
            if video_path[-4] != ".":
                video_path = video_path + ".mp4"
            else:
                raise ValueError(f"video file format is not verified: {video_path}")

        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        if frame_idx is not None:
            # using frame idx to sample frames
            start_frame_ind, end_frame_ind = frame_idx.split(":")
            start_frame_ind, end_frame_ind = int(start_frame_ind), int(start_frame_ind) + self.num_frames
            batch_index = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
        else:
            clip_length = min(video_length, (self.num_frames - 1) + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.num_frames, dtype=int)

        if video_path.endswith(".gif"):
            pixel_values = video_reader[batch_index]  # shape: (f, h, w, c)
        else:
            pixel_values = video_reader.get_batch(batch_index).asnumpy()  # shape: (f, h, w, c)
        return pixel_values, video_reader

    def get_text_data_and_mask(self, data_item):
        if not self.return_text_emb:
            text = data_item["cap"]
            text = self.text_preprocessing(text)
            text_tokens_and_mask = self.tokenizer(
                text,
                max_length=self.model_max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors=None,
            )
            input_ids = np.array(text_tokens_and_mask["input_ids"])
            cond_mask = np.array(text_tokens_and_mask["attention_mask"])
            if len(input_ids.shape) == 1:
                input_ids = input_ids[None, :]
            if len(cond_mask.shape) == 1:
                cond_mask = cond_mask[None, :]
            return input_ids, cond_mask
        else:
            text_emb_path = data_item["text_embed_path"]
            text_emb, cond_mask = self.parse_text_emb(text_emb_path)
            return text_emb, cond_mask

    def get_video(self, idx):
        video_path = self.vid_cap_list[idx]["path"]
        frame_idx = self.vid_cap_list[idx].get("frame_idx", None)
        if not self.return_vae_latent:
            video, video_reader = self.video_read(video_path, frame_idx)
            del video_reader
        else:
            raise NotImplementedError
            # video = self.parse_video_latent(vae_latent_path)
        video = self.apply_transform(video)
        video = video.transpose(3, 0, 1, 2)  # (f h w c) -> (c f h w)

        text_data, cond_mask = self.get_text_data_and_mask(self.vid_cap_list[idx])
        return video, text_data, cond_mask

    def get_image_from_video(self, video, text_data, cond_mask):
        select_image_idx = np.linspace(0, self.num_frames - 1, self.use_image_num, dtype=int)
        assert self.num_frames >= self.use_image_num
        image = [video[:, i : i + 1] for i in select_image_idx]  # num_img [c, 1, h, w]
        image = np.concatenate(image, axis=1)  # [c num_img h w]
        text_data = text_data.repeat(self.use_image_num, axis=0)  # self.use_image_num, l
        cond_mask = cond_mask.repeat(self.use_image_num, axis=0)  # self.use_image_num, l
        return image, text_data, cond_mask

    def get_image(self, idx):
        idx = idx % len(self.img_cap_list)  # out of range
        image_data = self.img_cap_list[idx]  # [{'path': path, 'cap': cap}, ...]

        image = [Image.open(i["path"]).convert("RGB") for i in image_data]  # num_img [h, w, c]
        image = [np.array(i)[None, ...] for i in image]  # num_img [1, h, w, c]
        image = [self.apply_transform(i) for i in image]  # num_img [1 H W C] -> num_img [1 H W C]
        image = [i.transpose(3, 0, 1, 2) for i in image]  # num_img [1 H W C] -> num_img [C 1 H W]
        image = np.concatenate(image, axis=1)  # [C num_img H W]

        text_data = []
        cond_mask = []
        for item in self.img_cap_list[idx]:
            t, m = self.get_text_data_and_mask(item)
            text_data.append(t)
            cond_mask.append(m)
        text_data = np.concatenate(text_data, axis=0)  # num_img, L, (D)
        cond_mask = np.concatenate(cond_mask, axis=0)  # num_img, L
        return image, text_data, cond_mask

    def get_batch(self, idx):
        # get video raw pixels (batch of frame) and its caption
        if self.num_frames != 1:
            video_data = self.get_video(idx)
            if self.use_image_num != 0:
                if self.use_img_from_vid:
                    image_data = self.get_image_from_video(*video_data)
                else:
                    image_data = self.get_image(idx)
                video = np.concatenate([video_data[0], image_data[0]], axis=1)  # c, num_frame+num_img, h, w
                text_data = np.concatenate([video_data[1], image_data[1]], axis=0)  # 1+num_img, L, (D)
                mask = np.concatenate([video_data[2], image_data[2]], axis=0)  # 1+self.use_image_num, l
                video_data = video, text_data, mask
            return video_data
        else:
            image_data = self.get_image(idx)  # 1 frame video as image
            return image_data

    def __len__(self):
        if self.num_frames != 1:
            return len(self.vid_cap_list)
        else:
            return len(self.img_cap_list)

    def apply_transform(self, pixel_values):
        # pixel value: (f, h, w, 3) -> transforms -> (f, h, w, 3)
        if self.transform_backend == "al":
            inputs = {"image": pixel_values[0]}
            num_frames = len(pixel_values)
            for i in range(num_frames - 1):
                inputs[f"image{i}"] = pixel_values[i + 1]
            output = self.pixel_transforms(**inputs)
            pixel_values = np.stack(list(output.values()), axis=0)
        else:
            raise NotImplementedError
        return pixel_values

    def __getitem__(self, idx):
        try:
            pixel_values, text, mask = self.get_batch(idx)
            if (self.prev_ok_sample is None) or (self.require_update_prev):
                self.prev_ok_sample = copy.deepcopy((pixel_values, text, mask))
                self.require_update_prev = False
        except Exception as e:
            logger.warning(f"Fail to get sample of idx {idx}. The corrupted video will be replaced.")
            print("\tError msg: {}".format(e), flush=True)
            assert self.prev_ok_sample is not None
            pixel_values, text, mask = self.prev_ok_sample  # unless the first sample is already not ok
            self.require_update_prev = True

            if idx >= len(self):
                raise IndexError  # needed for checking the end of dataset iteration
        if not self.return_vae_latent:
            # apply normalization
            pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)

        return pixel_values.astype(np.float32), text.astype(np.float32), mask.astype(np.int32)

    def parse_dataset_text(self, text_file):
        with open(text_file, "r") as f:
            folder_anno = [i.strip().split(",") for i in f.readlines() if len(i.strip()) > 0]
        if self.return_text_emb:
            expect_num = 3
        else:
            expect_num = 2
        data = []
        for item in folder_anno:
            if len(item) != expect_num:
                logger.warning(
                    f"Incorrect input text file! "
                    f"Expect to have {expect_num} paths, but got {len(item)} paths! "
                    "The input paths should be: video/image folder, text_embed_folder(optional), annotation file."
                )
                raise ValueError
            else:
                item_dict = dict(folder=item[0], annotation=item[-1])
                if expect_num == 3:
                    item_dict.update(dict(text_embed_folder=item[1]))
                data.append(item_dict)
        return data

    def get_text_embed_file_path(self, item):
        file_path = item["path"]
        # extra keys are identifiers added to the original file path
        for key in item.keys():
            if key not in ["cap", "path"]:
                identifer = f"-{key}-{item[key]}"
                file_path = Path(str(file_path))
                extension = file_path.suffix
                file_path = str(file_path.with_suffix("")) + identifer
                file_path = file_path + extension
        return Path(str(file_path)).with_suffix(".npz")

    def get_vid_cap_list(self):
        vid_cap_lists = []
        video_dataset = self.parse_dataset_text(self.video_data)
        assert len(video_dataset) > 0, f"The video dataset {self.video_data} must not be empty!"

        for item in video_dataset:
            anno = item["annotation"]
            with open(anno, "r") as f:
                vid_cap_list = json.load(f)
            logger.info(f"Building {anno}...")
            folder = item["folder"]
            text_embed_folder = item["text_embed_folder"] if self.return_text_emb else None

            new_vid_cap_list = []
            filtered_samples = 0
            for i in tqdm(range(len(vid_cap_list))):
                path = os.path.join(folder, vid_cap_list[i]["path"])
                text_embed_path = (
                    os.path.join(text_embed_folder, self.get_text_embed_file_path(vid_cap_list[i]))
                    if self.return_text_emb
                    else None
                )
                if os.path.exists(path.replace(".mp4", "_resize_1080p.mp4")):
                    path = path.replace(".mp4", "_resize_1080p.mp4")
                vid_cap_list[i]["path"] = path  # update the video path
                if text_embed_path is not None:
                    vid_cap_list[i]["text_embed_path"] = text_embed_path
                if self.filter_nonexistent:
                    if os.path.exists(path):
                        if not self.return_text_emb:
                            new_vid_cap_list.append(vid_cap_list[i])
                        else:
                            if os.path.exists(vid_cap_list[i]["text_embed_path"]):
                                new_vid_cap_list.append(vid_cap_list[i])
                            else:
                                filtered_samples += 1
                    else:
                        filtered_samples += 1
                else:
                    new_vid_cap_list.append(vid_cap_list[i])
            vid_cap_lists += new_vid_cap_list
        if self.filter_nonexistent:
            logger.info(f"Number of filtered video samples :{filtered_samples}")
        return vid_cap_lists

    def get_img_cap_list(self):
        use_image_num = self.use_image_num if self.use_image_num != 0 else 1
        img_cap_lists = []
        image_dataset = self.parse_dataset_text(self.image_data)
        filtered_samples = 0
        for item in image_dataset:
            anno = item["annotation"]
            with open(anno, "r") as f:
                img_cap_list = json.load(f)
            logger.info(f"Building {anno}...")
            folder = item["folder"]
            text_embed_folder = item["text_embed_folder"] if self.return_text_emb else None

            new_img_cap_list = []
            for i in tqdm(range(len(img_cap_list))):
                text_embed_path = (
                    os.path.join(text_embed_folder, self.get_text_embed_file_path(img_cap_list[i]))
                    if self.return_text_emb
                    else None
                )
                img_cap_list[i]["path"] = os.path.join(folder, img_cap_list[i]["path"])

                if text_embed_path is not None:
                    img_cap_list[i]["text_embed_path"] = text_embed_path
                if self.filter_nonexistent:
                    if os.path.exists(img_cap_list[i]["path"]):
                        if not self.return_text_emb:
                            new_img_cap_list.append(img_cap_list[i])
                        else:
                            if os.path.exists(img_cap_list[i]["text_embed_path"]):
                                new_img_cap_list.append(img_cap_list[i])
                            else:
                                filtered_samples += 1
                    else:
                        filtered_samples += 1
                else:
                    new_img_cap_list.append(img_cap_list[i])
            img_cap_lists += new_img_cap_list
        if self.filter_nonexistent:
            logger.info(f"Number of filtered image samples :{filtered_samples}")
        img_cap_lists = [img_cap_lists[i : i + use_image_num] for i in range(0, len(img_cap_lists), use_image_num)]
        return img_cap_lists[:-1]  # drop last to avoid error length


def create_dataloader(
    ds_config,
    batch_size,
    ds_name="text_video",
    num_parallel_workers=12,
    max_rowsize=64,
    shuffle=True,
    device_num=1,
    rank_id=0,
    drop_remainder=True,
    return_dataset=False,
    prefetch_size=None,
):
    if ds_name == "text_video":
        dataset = TextVideoDataset(**ds_config)
        column_names = ["video", "text", "mask"]
    else:
        raise NotImplementedError

    if prefetch_size is not None:
        assert isinstance(prefetch_size, int)
        ms.dataset.config.set_prefetch_size(prefetch_size)

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
    if return_dataset:
        return dl, dataset
    return dl
