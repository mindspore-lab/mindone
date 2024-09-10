import logging
from collections import defaultdict
from pprint import pformat
from typing import Tuple

import numpy as np

from .aspect import ASPECT_RATIOS
from .bucket import Bucket
from .video_dataset_refactored import VideoDatasetRefactored

_logger = logging.getLogger(__name__)


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


class BucketDistributedSampler:
    def __init__(
        self,
        dataset: VideoDatasetRefactored,
        buckets: Bucket,
        num_shards: int,
        shard_id: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_remainder: bool = True,
        verbose: bool = False,
    ):
        self._buckets = buckets
        self.verbose = verbose
        self.last_micro_batch_access_index = 0
        self.approximate_num_batch = None

        self._dataset = dataset

        self._seed = seed
        self._drop_remainder = drop_remainder
        self._shuffle = shuffle
        self._num_shards = num_shards
        self._shard_id = shard_id

        self.bucket_sample_dict = self.group_by_bucket()

    def __iter__(self) -> Tuple[int, int]:
        bucket_sample_dict = self.bucket_sample_dict

        rng = np.random.default_rng(self._seed)  # TODO: add epoch
        bucket_micro_batch_count = {}
        bucket_last_consumed = {}

        # process the samples
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle droplast
            bs_per_npu = self._buckets.get_batch_size(bucket_id)
            remainder = len(data_list) % bs_per_npu

            if remainder > 0:
                if not self._drop_remainder:
                    # if there is a remainder, we pad to make it divisible
                    data_list += data_list[: bs_per_npu - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            bucket_sample_dict[bucket_id] = data_list

            # handle shuffle
            if self._shuffle:
                data_list = rng.permutation(data_list).tolist()
                bucket_sample_dict[bucket_id] = data_list

            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // bs_per_npu
            bucket_micro_batch_count[bucket_id] = num_micro_batches

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in bucket_micro_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self._shuffle:
            bucket_id_access_order_indices = rng.permutation(len(bucket_id_access_order)).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # make the number of bucket accesses divisible by dp size
        remainder = len(bucket_id_access_order) % self._num_shards
        if remainder > 0:
            if self._drop_remainder:
                bucket_id_access_order = bucket_id_access_order[: len(bucket_id_access_order) - remainder]
            else:
                bucket_id_access_order += bucket_id_access_order[: self._num_shards - remainder]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order) // self._num_shards
        start_iter_idx = self.last_micro_batch_access_index // self._num_shards

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of NPUs
        self.last_micro_batch_access_index = start_iter_idx * self._num_shards
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self._buckets.get_batch_size(bucket_id)
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            bucket_access_list = bucket_id_access_order[i * self._num_shards : (i + 1) * self._num_shards]
            self.last_micro_batch_access_index += self._num_shards

            # compute the data samples consumed by each access
            bucket_access_boundaries = []
            for bucket_id in bucket_access_list:
                bucket_bs = self._buckets.get_batch_size(bucket_id)
                last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
                bucket_access_boundaries.append([last_consumed_index, last_consumed_index + bucket_bs])

                # update consumption
                if bucket_id in bucket_last_consumed:
                    bucket_last_consumed[bucket_id] += bucket_bs
                else:
                    bucket_last_consumed[bucket_id] = bucket_bs

            # compute the range of data accessed by each NPU
            bucket_id = bucket_access_list[self._shard_id]
            boundary = bucket_access_boundaries[self._shard_id]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0] : boundary[1]]
            for idx in cur_micro_batch:
                yield idx, bucket_id

        self.bucket_sample_dict = self.group_by_bucket()  # prepare samples for the next epoch
        self.reset()

    def __len__(self) -> int:
        return self.get_num_batch() // self._num_shards

    def group_by_bucket(self) -> dict:
        bucket_sample_dict = {}

        _logger.debug("Building buckets...")
        bucket_ids = [
            self._buckets.get_bucket_id(
                int(item["length"]),
                int(item["height"]),
                int(item["width"]),
                frame_interval=self._dataset._stride,
                seed=self._seed,  # TODO: incorrect seed, fix
            )
            for item in self._dataset._data
        ]

        # group by bucket
        # each data sample is put into a bucket with a similar image/video size
        for i in range(len(self._dataset)):
            bucket_id = bucket_ids[i]
            if bucket_id is None:
                continue
            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []
            bucket_sample_dict[bucket_id].append(i)
        return bucket_sample_dict

    def get_num_batch(self) -> int:
        bucket_sample_dict = self.group_by_bucket()
        self._get_num_batch_cached_bucket_sample_dict = bucket_sample_dict

        # calculate the number of batches
        if self.verbose:
            self._print_bucket_info(bucket_sample_dict)
        return self.approximate_num_batch

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        # collect statistics
        total_samples = 0
        total_batch = 0
        num_aspect_dict = defaultdict(lambda: [0, 0])
        num_hwt_dict = defaultdict(lambda: [0, 0])
        for k, v in bucket_sample_dict.items():
            size = len(v)
            num_batch = size // self._buckets.get_batch_size(k[:-1])

            total_samples += size
            total_batch += num_batch

            num_aspect_dict[k[-1]][0] += size
            num_aspect_dict[k[-1]][1] += num_batch
            num_hwt_dict[k[:-1]][0] += size
            num_hwt_dict[k[:-1]][1] += num_batch

        # sort
        num_aspect_dict = dict(sorted(num_aspect_dict.items(), key=lambda x: x[0]))
        num_hwt_dict = dict(
            sorted(num_hwt_dict.items(), key=lambda x: (ASPECT_RATIOS[x[0][0]][0], x[0][1]), reverse=True)
        )
        num_hwt_img_dict = {k: v for k, v in num_hwt_dict.items() if k[1] == 1}
        num_hwt_vid_dict = {k: v for k, v in num_hwt_dict.items() if k[1] > 1}

        # log
        if self._shard_id == 0 and self.verbose:
            _logger.info("Bucket Info:")
            _logger.info("Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(num_aspect_dict, sort_dicts=False))
            _logger.info("Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_img_dict, sort_dicts=False))
            _logger.info("Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_vid_dict, sort_dicts=False))
            _logger.info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                format_numel_str(total_batch),
                format_numel_str(total_samples),
                len(bucket_sample_dict),
            )
        self.approximate_num_batch = total_batch

    def reset(self):
        self.last_micro_batch_access_index = 0
