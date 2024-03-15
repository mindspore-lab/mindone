import logging
import os.path as osp

import numpy as np

logger = logging.getLogger()


class NPZEmbeddingCacheWriter:
    def __init__(self, cache_folder, start_index):
        self.cache_folder = cache_folder
        assert osp.exists(cache_folder)
        self.start_index = start_index
        self.index = start_index
        self.num_saved_files = 0
        self.num_failed_files = 0

    def save(self, save_name, save_dict):
        video_path = osp.join(self.cache_folder, save_name + ".npz")
        try:
            np.savez_compressed(video_path, **save_dict)
            self.num_saved_files += 1
        except Exception:
            logger.info(f"Failed to save {self.index}th video to path {video_path}")
            logger.info("Skipped saving it and continue...")
            self.num_failed_files += 1
        self.index += 1

    def get_status(self):
        logger.info(
            "NPZ embedding cache writer status:\n"
            f"Start Index: {self.start_index}.\n"
            f"Saving Attempts: {self.index}: save {self.num_saved_files} files, failed {self.num_failed_files} files.\n"
        )
