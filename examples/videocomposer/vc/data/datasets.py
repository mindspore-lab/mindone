import logging
import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from vc.annotator.mask import make_irregular_mask, make_rectangle_mask, make_uncrop
from vc.annotator.motion import extract_motion_vectors

__all__ = [
    "VideoDataset",
]

_logger = logging.getLogger(__name__)


class VideoDataset(object):
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
    ):
        """
        Args:
            root_dir: dir containing csv file which records video path and caption.
        """

        self.cfg = cfg

        self.max_words = max_words
        self.feature_framerate = feature_framerate
        self.max_frames = max_frames
        self.image_resolution = image_resolution
        self.transforms = transforms
        self.mv_transforms = mv_transforms
        self.misc_transforms = misc_transforms
        self.vit_transforms = vit_transforms
        self.vit_image_size = vit_image_size
        self.misc_size = misc_size
        self.mvs_visual = mvs_visual

        if root_dir is not None:
            video_paths, captions = get_video_paths_captions(root_dir)
            num_samples = len(video_paths)
            self.video_cap_pairs = [[video_paths[i], captions[i]] for i in range(num_samples)]
        else:
            self.video_cap_pairs = [[self.cfg.input_video, self.cfg.input_text_desc]]

    def __len__(self):
        return len(self.video_cap_pairs)

    def __getitem__(self, index):
        video_key, cap_txt = self.video_cap_pairs[index]

        feature_framerate = self.feature_framerate
        if os.path.exists(video_key):
            ref_frame, vit_image, video_data, misc_data, mv_data = self._get_video_train_data(
                video_key, feature_framerate, self.mvs_visual
            )
        else:  # use dummy data
            _logger.warning(f"The video: {video_key} does not exist! Please check the video path.")
            ref_frame, vit_image, video_data, misc_data, mv_data = self._get_dummy_data(video_key)

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

        return ref_frame, cap_txt, video_data, misc_data, feature_framerate, mask, mv_data

    def _get_dummy_data(self, video_key):
        ref_frame = np.zeros((3, self.vit_image_size, self.vit_image_size), dtype=np.float32)
        vit_image = np.zeros((3, self.vit_image_size, self.vit_image_size), dtype=np.float32)  # noqa
        video_data = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float32)
        misc_data = np.zeros((self.max_frames, 3, self.misc_size, self.misc_size), dtype=np.float32)
        mv_data = np.zeros((self.max_frames, 2, self.image_resolution, self.image_resolution), dtype=np.float32)

        return ref_frame, vit_image, video_data, misc_data, mv_data

    def _get_video_train_data(self, video_key, feature_framerate, viz_mv):
        filename = video_key
        frame_types, frames, mvs, mvs_visual = extract_motion_vectors(
            input_video=filename, fps=feature_framerate, viz=viz_mv
        )

        total_frames = len(frame_types)
        start_indices = np.where(
            (np.array(frame_types) == "I") & (total_frames - np.arange(total_frames) >= self.max_frames)
        )[0]

        if start_indices.size == 0:  # empty, no frames
            _logger.warning(f"Failed to load the video: {filename}. The video may be broken.")
            return self._get_dummy_data(filename)

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

        video_data = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float32)
        mv_data = np.zeros((self.max_frames, 2, self.image_resolution, self.image_resolution), dtype=np.float32)
        misc_data = np.zeros((self.max_frames, 3, self.misc_size, self.misc_size), dtype=np.float32)
        if have_frames:
            video_data[: len(frames), ...] = frames
            misc_data[: len(frames), ...] = misc_imgs
            mv_data[: len(frames), ...] = mvs

        ref_frame = vit_image

        return ref_frame, vit_image, video_data, misc_data, mv_data


def get_video_paths_captions(data_dir):
    anno_list = sorted(
        [os.path.join(data_dir, f) for f in list(filter(lambda x: x.endswith(".csv"), os.listdir(data_dir)))]
    )
    db_list = [pd.read_csv(f) for f in anno_list]
    video_paths = []
    all_captions = []
    for db in db_list:
        video_paths.extend(list(db["video"]))
        all_captions.extend(list(db["caption"]))
    assert len(video_paths) == len(all_captions)
    video_paths = [os.path.join(data_dir, f) for f in video_paths]
    # _logger.info(f"Before filter, Total number of training samples: {len(video_paths)}")

    return video_paths, all_captions
