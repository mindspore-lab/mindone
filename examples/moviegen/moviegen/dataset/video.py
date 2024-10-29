from typing import Tuple

import numpy as np


class VideoDataset:
    def __len__(self) -> int:
        return NotImplementedError()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            video/video caching
            text embedding 1
            text embedding 1
            text embedding 1
        """
        raise NotImplementedError()
