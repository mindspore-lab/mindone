import json
import logging
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from vc.annotator.mask import make_irregular_mask, make_rectangle_mask, make_uncrop
from vc.annotator.motion import extract_motion_vectors

from mindspore import dataset as ds

from .transforms import create_transforms

__all__ = [
    "VideoDatasetForTrain",
]

_logger = logging.getLogger(__name__)


def make_masked_image(img, mask):
    masked_img = np.concatenate([img * (1 - mask), (1 - mask)], axis=1)
    return masked_img


class VideoDatasetForTrain(object):
    def __init__(
        self,
        cfg=None,
        root_dir=None,
        max_words=30,
        feature_framerate=1,
        max_frames=16,
        image_resolution=224,
        transforms=None,
        mv_transforms=None,
        misc_transforms=None,
        vit_transforms=None,
        vit_image_size=336,
        misc_size=384,
        mvs_visual=False,
        tokenizer=None,
        conditions_for_train=None,
        record_data_stat=False,
        rank_id=0,
    ):
        """
        Args:
            root_dir: dir containing csv file which records video path and caption.
        """

        self.cfg = cfg

        self.tokenizer = tokenizer
        self.max_words = max_words
        self.feature_framerate = feature_framerate
        self.max_frames = max_frames
        self.image_resolution = image_resolution
        self.transforms = transforms
        self.mv_transforms = mv_transforms
        self.misc_transforms = misc_transforms
        self.vit_transforms = vit_transforms
        self.canny_detector = None  # canny is not used by UNetSD
        self.vit_image_size = vit_image_size
        self.misc_size = misc_size
        self.mvs_visual = mvs_visual
        self.conditions_for_train = conditions_for_train

        if root_dir is not None:
            video_paths, captions = get_video_paths_captions(root_dir)
            num_samples = len(video_paths)
            self.video_cap_pairs = [[video_paths[i], captions[i]] for i in range(num_samples)]
        else:
            self.video_cap_pairs = [[self.cfg.input_video, self.cfg.input_text_desc]]

        self.tokenizer = tokenizer  # bpe

        self.record_data_stat = record_data_stat
        if self.record_data_stat:
            header = ",".join(["video_path", "frames", "resolution", "load_time"])
            self.stat_fp = os.path.join(cfg.output_dir, f"data_stat_rank_{rank_id}.csv")
            with open(self.stat_fp, "w", encoding="utf-8") as fp:
                fp.write(header + "\n")

    def tokenize(self, text, max_length=77):
        tokens = self.tokenizer(text, padding="max_length", max_length=max_length)["input_ids"]
        # truncate for over-long tokens
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            tokens[-1] = self.tokenizer.eos_token_id

        return tokens

    def __len__(self):
        return len(self.video_cap_pairs)

    def __getitem__(self, index):
        video_key, cap_txt = self.video_cap_pairs[index]

        feature_framerate = self.feature_framerate
        if os.path.exists(video_key):
            try:
                vit_image, video_data, misc_data, mv_data = self._get_video_train_data(
                    video_key, feature_framerate, self.mvs_visual
                )
            except Exception as e:
                print("Load video {} fails, Error: {}".format(video_key, e), flush=True)
                _logger.warning(
                    f"Fail to load {video_key}, video data could be broken, which will be replaced with dummy data."
                )
                vit_image, video_data, misc_data, mv_data = self._get_dummy_data(video_key)
        else:  # use dummy data
            _logger.warning(
                f"Fail to load {video_key}, video data could be broken, which will be replaced with dummy data."
            )
            vit_image, video_data, misc_data, mv_data = self._get_dummy_data(video_key)

        # inpainting mask
        p = random.random()
        if p < 0.7:
            mask = make_irregular_mask(512, 512)
        elif p < 0.9:
            mask = make_rectangle_mask(512, 512)
        else:
            mask = make_uncrop(512, 512)
        mask = cv2.resize(mask, (self.misc_size, self.misc_size), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=0)
        mask = np.repeat(mask, repeats=self.max_frames, axis=0)
        masked_video = make_masked_image((misc_data - 0.5) / 0.5, mask)  # (f, c, h, w)

        # adapt for training, output element must map the order of model construct input
        caption_tokens = self.tokenize(cap_txt)
        # style_image = vit_image
        single_image = misc_data[:1].copy()  # [1, 3, h, w]

        return (
            video_data,
            caption_tokens,
            feature_framerate,
            vit_image,
            mv_data,
            single_image,
            masked_video,
            misc_data,
        )

    def _get_dummy_data(self, video_key):
        _logger.warning(f"The video path: {video_key} does not exist or no video dir provided!")
        vit_image = np.zeros((3, self.vit_image_size, self.vit_image_size), dtype=np.float32)  # noqa
        video_data = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float32)
        misc_data = np.zeros((self.max_frames, 3, self.misc_size, self.misc_size), dtype=np.float32)
        mv_data = np.zeros((self.max_frames, 2, self.image_resolution, self.image_resolution), dtype=np.float32)

        return vit_image, video_data, misc_data, mv_data

    def _get_video_train_data(self, video_key, feature_framerate, viz_mv):
        filename = video_key
        if self.record_data_stat:
            vstart = time.time()

        frame_types, frames, mvs, mvs_visual = extract_motion_vectors(
            input_video=filename, fps=feature_framerate, viz=viz_mv
        )

        if self.record_data_stat:
            _raw_frames_len = len(frames) * 4
            _resolution = frames[0].shape[-3:-1]
            _stat = f"{video_key},{_raw_frames_len},{_resolution},{time.time()-vstart}"
            with open(self.stat_fp, "a", encoding="utf-8") as fp:
                fp.write(_stat + "\n")

        total_frames = len(frame_types)
        start_indices = np.where(
            (np.array(frame_types) == "I") & (total_frames - np.arange(total_frames) >= self.max_frames)
        )[0]

        start_index = np.random.choice(start_indices)
        indices = np.arange(start_index, start_index + self.max_frames)

        # note frames are in BGR mode, need to trans to RGB mode
        frames = [Image.fromarray(frames[i][:, :, ::-1]) for i in indices]
        mvs = [mvs[i].astype(np.float32) for i in indices]  # h, w, 2

        have_frames = len(frames) > 0
        middle_index = int(len(frames) / 2)
        if have_frames:
            ref_frame = frames[middle_index]
            vit_image = self.vit_transforms(ref_frame)[0]
            misc_imgs = np.stack([self.misc_transforms(frame)[0] for frame in frames], axis=0)
            frames = np.stack([self.transforms(frame)[0] for frame in frames], axis=0)
            mvs = np.stack([self.mv_transforms(mv).transpose((2, 0, 1)) for mv in mvs], axis=0)
        else:
            raise RuntimeError(f"Got no frames from {filename}!")
            # vit_image = np.zerso(3, self.vit_image_size, self.vit_image_size)

        video_data = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float32)
        mv_data = np.zeros((self.max_frames, 2, self.image_resolution, self.image_resolution), dtype=np.float32)
        misc_data = np.zeros((self.max_frames, 3, self.misc_size, self.misc_size), dtype=np.float32)
        if have_frames:
            video_data[: len(frames), ...] = frames
            misc_data[: len(frames), ...] = misc_imgs
            mv_data[: len(frames), ...] = mvs

        return vit_image, video_data, misc_data, mv_data


def get_video_paths_captions(data_dir, only_use_csv_anno=False):
    """
    JSON files have higher priority, i.e., if both JSON and csv annotion files exist, only JSON files will be loaded.
    To force to read CSV annotation, please parse only_use_csv_anno=True.
    """
    csv_anno_list = sorted(
        [os.path.join(data_dir, f) for f in list(filter(lambda x: x.endswith(".csv"), os.listdir(data_dir)))]
    )
    json_anno_list = sorted(
        [os.path.join(data_dir, f) for f in list(filter(lambda x: x.endswith(".json"), os.listdir(data_dir)))]
    )

    video_paths = []
    all_captions = []
    if (len(json_anno_list) == 0) or only_use_csv_anno:
        _logger.info("Reading annotation from csv files: {}".format(csv_anno_list))
        db_list = [pd.read_csv(f) for f in csv_anno_list]
        for db in db_list:
            video_paths.extend(list(db["video"]))
            all_captions.extend(list(db["caption"]))
        # _logger.info(f"Before filter, Total number of training samples: {len(video_paths)}")
    elif len(json_anno_list) > 0:
        _logger.info("Reading annotation from json files: {}".format(json_anno_list))
        for json_fp in json_anno_list:
            with open(json_fp, "r", encoding="utf-8") as fp:
                datasets_dict = json.load(fp)
                for dataset in datasets_dict:
                    rel_path_caption_pair_list = datasets_dict[dataset]
                    for rel_path_caption_pair in rel_path_caption_pair_list:
                        video_paths.append(rel_path_caption_pair[0])
                        all_captions.append(rel_path_caption_pair[1])

    assert len(video_paths) == len(all_captions)
    video_paths = [os.path.join(data_dir, f) for f in video_paths]
    # print("D--: ", video_paths, all_captions)

    return video_paths, all_captions


def build_dataset(cfg, device_num, rank_id, tokenizer, record_data_stat=False):
    infer_transforms, misc_transforms, mv_transforms, vit_transforms = create_transforms(cfg)
    dataset = VideoDatasetForTrain(
        cfg=cfg,
        root_dir=cfg.root_dir,
        max_words=cfg.max_words,
        feature_framerate=cfg.feature_framerate,
        max_frames=cfg.max_frames,
        image_resolution=cfg.resolution,
        transforms=infer_transforms,
        mv_transforms=mv_transforms,
        misc_transforms=misc_transforms,
        vit_transforms=vit_transforms,
        vit_image_size=cfg.vit_image_size,
        misc_size=cfg.misc_size,
        mvs_visual=cfg.mvs_visual,
        tokenizer=tokenizer,
        record_data_stat=record_data_stat,
        rank_id=rank_id,
    )

    print("Total number of samples: ", len(dataset))

    dataloader = ds.GeneratorDataset(
        source=dataset,
        column_names=[
            "video_data",
            "caption_tokens",
            "feature_framerate",
            "vit_image",
            "mv_data",
            "single_image",
            "masked",
            "misc_data",
        ],
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=cfg.shuffle,
        num_parallel_workers=cfg.num_parallel_workers,
        max_rowsize=128,  # video data require larger rowsize
    )

    dl = dataloader.batch(
        cfg.batch_size,
        drop_remainder=True,
    )

    return dl
