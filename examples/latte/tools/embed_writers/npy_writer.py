import logging
import os
import os.path as osp

import numpy as np

logger = logging.getLogger()


class NPYEmbeddingCacheWriter:
    def __init__(self, cache_folder, start_index):
        self.cache_folder = cache_folder
        assert osp.exists(cache_folder)
        self.start_index = start_index
        self.index = start_index
        self.num_saved_files = 0
        self.num_failed_files = 0

    def save(self, save_video_name, frames_names, save_dict):
        video_folder = osp.join(self.cache_folder, save_video_name)
        if not osp.exists(video_folder):
            os.makedirs(video_folder)

        video_latent = save_dict["video_latent"]
        assert len(frames_names) == len(video_latent)
        try:
            for frame_name, latent in zip(frames_names, video_latent):
                file_path = osp.join(video_folder, f"{frame_name}.npy")
                np.save(file_path, latent)

            # for other key arguments
            for key in save_dict:
                if key != "video_latent":
                    file_path = osp.join(video_folder, f"{key}.npy")
                    np.save(file_path, save_dict[key])
            self.num_saved_files += 1
        except Exception as e:
            logger.info(f"Failed to save for video {save_video_name}: {len(frames_names)} frames")
            print(e)
            self.num_failed_files += 1
