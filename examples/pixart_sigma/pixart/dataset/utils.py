from typing import Callable, Dict, List, Tuple

import numpy as np

__all__ = ["classify_height_width_bin", "bucket_split_function"]


def classify_height_width_bin(height: int, width: int, ratios: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Returns binned height and width."""
    ar = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    default_hw = ratios[closest_ratio]
    return default_hw[0], default_hw[1]


def bucket_split_function(
    ratios: Dict[str, Tuple[int, int]], batch_size: int
) -> Tuple[Callable[[np.ndarray], int], List[int], List[int]]:
    hashed_buckets = dict(zip(ratios.values(), range(1, len(ratios))))
    bucket_boundaries = list(hashed_buckets.values())
    bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)

    def _bucket_split_function(x: np.ndarray) -> int:
        # C H W
        _, H, W = x.shape
        return hashed_buckets[(H, W)]

    return _bucket_split_function, bucket_boundaries, bucket_batch_sizes
