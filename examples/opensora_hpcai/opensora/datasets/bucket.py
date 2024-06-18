"""
Credit: OpenSora HPC-AI Tech
https://github.com/hpcaitech/Open-Sora/blob/ea41df3d6cc5f389b6824572854d97fa9f7779c3/opensora/datasets/bucket.py
"""
from random import random

import numpy as np

from .aspect import ASPECT_RATIOS, get_closest_ratio


class Bucket:
    def __init__(self, bucket_config):
        for key in bucket_config:
            assert key in ASPECT_RATIOS, f"Aspect ratio {key} not found."
        bucket_probs, bucket_bs = {}, {}
        bucket_names = sorted(bucket_config.keys(), key=lambda x: ASPECT_RATIOS[x][0], reverse=True)
        for key in bucket_names:
            bucket_time_names = sorted(bucket_config[key].keys(), key=lambda x: x, reverse=True)
            bucket_probs[key] = {k: bucket_config[key][k][0] for k in bucket_time_names}
            bucket_bs[key] = {k: bucket_config[key][k][1] for k in bucket_time_names}

        # first level: HW
        num_bucket = 0
        hw_criteria = dict()
        t_criteria = dict()
        ar_criteria = dict()
        bucket_id = dict()
        bucket_id_cnt = 0
        for k1, v1 in bucket_probs.items():
            hw_criteria[k1] = ASPECT_RATIOS[k1][0]
            t_criteria[k1] = dict()
            ar_criteria[k1] = dict()
            bucket_id[k1] = dict()
            for k2, _ in v1.items():
                t_criteria[k1][k2] = k2
                bucket_id[k1][k2] = bucket_id_cnt
                bucket_id_cnt += 1
                ar_criteria[k1][k2] = dict()
                for k3, v3 in ASPECT_RATIOS[k1][1].items():
                    ar_criteria[k1][k2][k3] = v3
                    num_bucket += 1

        self.bucket_probs = bucket_probs
        self.bucket_bs = bucket_bs
        self.bucket_id = bucket_id
        self.hw_criteria = hw_criteria
        self.t_criteria = t_criteria
        self.ar_criteria = ar_criteria
        self.num_bucket = num_bucket

    def get_bucket_id(self, T, H, W, frame_interval=1):
        resolution = H * W
        approx = 0.8

        fail = True
        for hw_id, t_criteria in self.bucket_probs.items():
            if resolution < self.hw_criteria[hw_id] * approx:
                continue

            # if sample is an image
            if T == 1:
                if 1 in t_criteria:
                    if random() < t_criteria[1]:
                        fail = False
                        t_id = 1
                        break
                else:
                    continue

            # otherwise, find suitable t_id for video
            t_fail = True
            for t_id, prob in t_criteria.items():
                if isinstance(prob, list):
                    prob_t = prob[1]
                    if random() > prob_t:
                        continue
                if T > t_id * frame_interval and t_id != 1:
                    t_fail = False
                    break
            if t_fail:
                continue

            # leave the loop if prob is high enough
            if isinstance(prob, list):
                prob = prob[0]
            if prob >= 1 or random() < prob:
                fail = False
                break
        if fail:
            return None

        # get aspect ratio id
        ar_criteria = self.ar_criteria[hw_id][t_id]
        ar_id = get_closest_ratio(H, W, ar_criteria)
        return hw_id, t_id, ar_id

    def get_thw(self, bucket_id):
        assert len(bucket_id) == 3
        T = self.t_criteria[bucket_id[0]][bucket_id[1]]
        H, W = self.ar_criteria[bucket_id[0]][bucket_id[1]][bucket_id[2]]
        return T, H, W

    def get_prob(self, bucket_id):
        return self.bucket_probs[bucket_id[0]][bucket_id[1]]

    def get_batch_size(self, bucket_id):
        return self.bucket_bs[bucket_id[0]][bucket_id[1]]

    def __len__(self):
        return self.num_bucket


def bucket_split_function(buckets: Bucket):
    hashed_buckets = {}

    batch_sizes = []
    cnt = 1
    for name, lengths in buckets.ar_criteria.items():
        for length, ars in lengths.items():
            if buckets.bucket_bs[name][length] is not None and buckets.bucket_bs[name][length] > 0:
                for ar, (h, w) in ars.items():
                    hashed_buckets.setdefault(length, {}).setdefault(h, {})[w] = cnt
                    batch_sizes.append(buckets.bucket_bs[name][length])
                    cnt += 1

    def _bucket_split_function(video: np.ndarray) -> int:
        # video: F C H W
        return hashed_buckets[video.shape[0]][video.shape[2]][video.shape[3]]

    return _bucket_split_function, list(range(1, cnt - 1)), batch_sizes
