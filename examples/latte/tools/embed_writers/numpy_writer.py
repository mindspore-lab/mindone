import logging
import os
import os.path as osp

import numpy as np

logger = logging.getLogger()


class NumpyEmbeddingCacheWriter:
    def __init__(self, cache_folder, start_index, save_as_npy_keys=[], save_as_npz_keys=[]):
        self.cache_folder = cache_folder
        assert osp.exists(cache_folder)
        self.start_index = start_index
        self.index = start_index
        self.num_saved_files = 0
        self.num_failed_files = 0
        self.save_as_npy_keys = save_as_npy_keys
        self.save_as_npz_keys = save_as_npz_keys
        assert len(self.save_as_npy_keys) + len(self.save_as_npz_keys) != 0, "At least save something!"

    def save_npy_files(self, save_video_name, frames_names, save_dict):
        video_folder = osp.join(self.cache_folder, save_video_name)

        if not osp.exists(video_folder) and len(self.save_as_npy_keys) > 0:
            os.makedirs(video_folder)
        for npy_key in self.save_as_npy_keys:
            npy_value = save_dict[npy_key]
            assert len(frames_names) == len(npy_value), f"Expect {npy_key} value should match with the frame length"
            for frame_name, latent in zip(frames_names, npy_value):
                file_path = osp.join(video_folder, f"{npy_key}-{frame_name}.npy")
                np.save(file_path, latent)

    def save_npz_files(self, save_video_name, save_dict):
        save_npz_keys = []
        for npz_key in self.save_as_npz_keys:
            if npz_key in save_dict:
                save_npz_keys.append(npz_key)
        if len(save_npz_keys) > 0:
            file_path = osp.join(self.cache_folder, save_video_name + ".npz")
            np.savez_compressed(file_path, **dict([(k, save_dict[k]) for k in save_npz_keys]))

    def save(self, save_video_name, frames_names, save_dict):
        try:
            self.save_npy_files(save_video_name, frames_names, save_dict)
            self.save_npz_files(save_video_name, save_dict)
            self.num_saved_files += 1
        except Exception as e:
            logger.info(e)
            logger.info(f"Failed to save numpy embeddings, skip for {save_video_name}")
            self.num_failed_files += 1

    def get_status(self):
        logger.info(
            "Numpy embedding cache writer status:\n"
            f"Start Video Index: {self.start_index}.\n"
            f"Saving Attempts: {self.index}: save {self.num_saved_files} videos, failed {self.num_failed_files} videos.\n"
        )
