from math import sqrt
from typing import Callable, Dict, List, Tuple

import numpy as np

__all__ = ["get_target_size", "bucket_split_function"]

AR = {"1:1", "5:4", "4:3", "16:9", "16:10", "21:9", "2:1"}
RESOLUTIONS = {"256px": 256**2, "768px": 768**2}

# for internal use
_AR = {ar: int(ar.split(":")[0]) / int(ar.split(":")[1]) for ar in AR}  # horizontal
_AR.update({":".join(ar.split(":")[::-1]): int(ar.split(":")[1]) / int(ar.split(":")[0]) for ar in AR})  # vertical
_ARS = np.array(list(_AR.values()))  # convert to a numpy array once


def _get_resolutions() -> Dict[str, Dict[float, Tuple[int, int]]]:
    res = {}
    for name, pix_val in RESOLUTIONS.items():
        res[name] = {}
        for ar, ar_val in _AR.items():
            wr, hr = [int(a) for a in ar.split(":")]
            x = sqrt(pix_val / wr / hr)
            w, h = int(x * wr), int(x * hr)
            w, h = w - (w % 2), h - (h % 2)  # make sides even
            res[name][ar_val] = (h, w)
    return res


_RES = _get_resolutions()


def get_target_size(res: str, h: int, w: int) -> Tuple[int, int]:
    ar = _ARS[np.argmin(np.abs(_ARS - w / h))]  # find the closest matching AR
    return _RES[res][ar]  # noqa


def bucket_split_function(frames: Dict[int, int]) -> Tuple[Callable[[np.ndarray], int], List[int], List[int]]:
    hashed_buckets, batch_sizes, cnt = {}, [], 0

    for f, bs in frames.items():
        hashed_buckets[f] = {}
        for ar in _AR.values():
            hashed_buckets[f][ar] = cnt
            batch_sizes.append(bs)
            cnt += 1

    def _bucket_split_function(video: np.ndarray) -> int:
        # video: (T C H W)
        t, _, h, w = video.shape
        ar = _ARS[np.argmin(np.abs(_ARS - w / h))]  # find the closest matching AR
        return hashed_buckets[t][ar]

    return _bucket_split_function, list(range(1, cnt)), batch_sizes
