from typing import Callable, List, Tuple

import numpy as np


def bucket_split_function(
    image_batch_size: int, video_batch_size: int
) -> Tuple[Callable[[np.ndarray], int], List[int], List[int]]:
    return (
        lambda x: int(x.shape[0] > 1),  # image or video
        [1],  # 2 buckets for now: image and videos of fixed length
        [image_batch_size, video_batch_size],
    )
