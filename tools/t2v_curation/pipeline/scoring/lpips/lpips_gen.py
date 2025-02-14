import argparse
import os

import numpy as np
import pandas as pd

import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.communication import get_group_size, get_rank, init
from tqdm import tqdm

from pipeline.datasets.utils import extract_frames
from pipeline.scoring.lpips.lpips import LPIPS
# Commented out because we'll handle merging differently
# from pipeline.scoring.utils import merge_scores

def merge_scores(gathered_list: list, meta: pd.DataFrame, columns: list):
    """
    Merge multiple columns of scores into the meta DataFrame.

    Parameters:
    - gathered_list: list of tuples, each containing indices_list and multiple scores_lists
    - meta: pandas DataFrame to merge the scores into
    - columns: list of column names corresponding to the scores_lists
    """
    # Initialize empty lists for indices and scores
    flat_indices = []
    flat_scores_dict = {col: [] for col in columns}

    # Iterate over gathered_list
    for item in gathered_list:
        indices_list = item[0]
        scores_lists = item[1:]  # List of lists, one per column

        # Flatten indices
        flat_indices.extend(indices_list)

        # Flatten scores for each column
        for col_idx, col in enumerate(columns):
            flat_scores_dict[col].extend(scores_lists[col_idx])

    # Convert to numpy arrays
    flat_indices = np.array(flat_indices)
    for col in columns:
        flat_scores_dict[col] = np.array(flat_scores_dict[col])

    # Remove duplicates
    unique_indices, unique_indices_idx = np.unique(flat_indices, return_index=True)

    # Assign scores to meta DataFrame
    for col in columns:
        meta.loc[unique_indices, col] = flat_scores_dict[col][unique_indices_idx]

    # Keep only the rows with unique indices
    meta = meta.loc[unique_indices]
    return meta


class VideoTextDataset:
    def __init__(self, meta_path, seconds=1, target_size=(224, 224)):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.seconds = seconds
        self.target_size = target_size  # target reshape size (downsample for images)

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample["path"]

        images, timestamps = extract_frames(
            path, seconds=self.seconds, backend="decord", return_timestamps=True
        )
        images = [np.array(img.resize(self.target_size)) for img in images]
        images = np.stack(images)  # Shape: [N, H, W, C]
        images = images.transpose((0, 3, 1, 2))  # Shape: [N, C, H, W]
        timestamps = np.array(timestamps)  # Shape: [N]

        return index, images, timestamps

    def __len__(self):
        return len(self.meta)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--use_cpu", action="store_true", help="Whether to use CPU")
    # REMARK: batch size should be 1 unless all video clips are of the same length and the same resolution
    # we recommend keeping bs = 1
    parser.add_argument("--bs", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--seconds",
        type=int,
        default=1,
        help="Interval (in seconds) at which frames are sampled from the video.",
    )
    parser.add_argument("--target_height", type=int, default=224,
                        help="Target image height to be processed")
    parser.add_argument("--target_width", type=int, default=224,
                        help="Target image width to be processed")
    parser.add_argument("--skip_if_existing", action="store_true")
    parser.add_argument(
        "--lpips_ckpt_path",
        type=str,
        default="pretrained_models/lpips.ckpt",
        help="Load LPIPS model checkpoint.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_lpips{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    if not args.use_cpu:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        init()
        rank_id = get_rank()
        rank_size = get_group_size()

    model = LPIPS()
    model.load_from_pretrained(args.lpips_ckpt_path)
    model.set_train(False)

    dataset_generator = (
        VideoTextDataset(
            meta_path=args.meta_path,
            seconds=args.seconds,
            target_size=(args.target_height, args.target_width))
    )
    if not args.use_cpu:
        dataset = ds.GeneratorDataset(
            source=dataset_generator,
            column_names=["index", "images", "timestamps"],
            shuffle=False,
            num_shards=rank_size,
            shard_id=rank_id,
        )
    else:
        dataset = ds.GeneratorDataset(
            source=dataset_generator,
            column_names=["index", "images", "timestamps"],
            shuffle=False,
        )
    dataset = dataset.batch(args.bs, drop_remainder=False)
    iterator = dataset.create_dict_iterator(num_epochs=1)

    # Initialize lists to store results
    indices_list = []
    avg_scores_list = []       # (1) Average LPIPS score (weighted)
    avg_lpips_list = []        # (1) Average LPIPS score (unweighted)
    perc_lt_0_02_list = []     # (2) Percentage of LPIPS scores < 0.02
    perc_lt_0_1_list = []      # (3) Percentage of LPIPS scores < 0.1
    perc_lt_0_2_list = []      # (4) Percentage of LPIPS scores < 0.2
    perc_gt_avg_list = []      # (5) Percentage of LPIPS scores > average LPIPS score
    lpips_count_list = []      # (6) LPIPS score count (N-1)

    # Compute LPIPS scores
    for batch in tqdm(iterator):
        indices = batch["index"].asnumpy().tolist()
        images = batch["images"]
        timestamps = batch["timestamps"]

        for i in range(len(indices)):
            index = indices[i]
            imgs = images[i]
            times = timestamps[i].asnumpy()

            if imgs.shape[0] < 2:
                avg_score = -1.0
                avg_lpips = -1.0
                perc_lt_0_02 = -1.0
                perc_lt_0_1 = -1.0
                perc_lt_0_2 = -1.0
                perc_gt_avg = -1.0
                lpips_count = 0
            else:
                # Compute time differences between frames
                delta_times = times[1:] - times[:-1]  # Shape: [N-1]

                # Prepare pairs of frames
                imgs_0 = imgs[:-1]  # [N-1, C, H, W]
                imgs_1 = imgs[1:]   # [N-1, C, H, W]

                scores = model(imgs_0, imgs_1)  # [N-1, 1]
                scores = scores.squeeze().asnumpy()  # [N-1]

                # Weighted sum of scores
                weighted_scores = scores * delta_times
                total_time = times[-1] - times[0]
                avg_score = weighted_scores.sum() / total_time

                # Compute additional metrics
                avg_lpips = scores.mean()  # Unweighted average LPIPS score
                perc_lt_0_02 = (scores < 0.02).mean() * 100  # Percentage
                perc_lt_0_1 = (scores < 0.1).mean() * 100
                perc_lt_0_2 = (scores < 0.2).mean() * 100
                perc_gt_avg = (scores > avg_lpips).mean() * 100
                lpips_count = len(scores)  # N-1

            indices_list.append(index)
            avg_scores_list.append(avg_score)
            avg_lpips_list.append(avg_lpips)
            perc_lt_0_02_list.append(perc_lt_0_02)
            perc_lt_0_1_list.append(perc_lt_0_1)
            perc_lt_0_2_list.append(perc_lt_0_2)
            perc_gt_avg_list.append(perc_gt_avg)
            lpips_count_list.append(lpips_count)

    # AllGather results if necessary
    if not args.use_cpu:
        allgather = ops.AllGather()
        indices_tensor = Tensor(indices_list, ms.int64)
        avg_scores_tensor = Tensor(avg_scores_list, ms.float32)
        avg_lpips_tensor = Tensor(avg_lpips_list, ms.float32)
        perc_lt_0_02_tensor = Tensor(perc_lt_0_02_list, ms.float32)
        perc_lt_0_1_tensor = Tensor(perc_lt_0_1_list, ms.float32)
        perc_lt_0_2_tensor = Tensor(perc_lt_0_2_list, ms.float32)
        perc_gt_avg_tensor = Tensor(perc_gt_avg_list, ms.float32)
        lpips_count_tensor = Tensor(lpips_count_list, ms.int64)

        indices_list = allgather(indices_tensor).asnumpy().tolist()
        avg_scores_list = allgather(avg_scores_tensor).asnumpy().tolist()
        avg_lpips_list = allgather(avg_lpips_tensor).asnumpy().tolist()
        perc_lt_0_02_list = allgather(perc_lt_0_02_tensor).asnumpy().tolist()
        perc_lt_0_1_list = allgather(perc_lt_0_1_tensor).asnumpy().tolist()
        perc_lt_0_2_list = allgather(perc_lt_0_2_tensor).asnumpy().tolist()
        perc_gt_avg_list = allgather(perc_gt_avg_tensor).asnumpy().tolist()
        lpips_count_list = allgather(lpips_count_tensor).asnumpy().tolist()
    else:
        # Ensure lists are in the correct format
        indices_list = indices_list
        avg_scores_list = avg_scores_list
        avg_lpips_list = avg_lpips_list
        perc_lt_0_02_list = perc_lt_0_02_list
        perc_lt_0_1_list = perc_lt_0_1_list
        perc_lt_0_2_list = perc_lt_0_2_list
        perc_gt_avg_list = perc_gt_avg_list
        lpips_count_list = lpips_count_list

    # Prepare gathered_list for merge_scores
    gathered_list = [
        (
            indices_list,
            avg_scores_list,
            avg_lpips_list,
            perc_lt_0_02_list,
            perc_lt_0_1_list,
            perc_lt_0_2_list,
            perc_gt_avg_list,
            lpips_count_list,
        )
    ]

    columns = [
        "lpips_avg_weighted",
        "lpips_avg_unweighted",
        "lpips_perc_lt_0_02",
        "lpips_perc_lt_0_1",
        "lpips_perc_lt_0_2",
        "lpips_perc_gt_avg",
        "lpips_count",
    ]

    if args.use_cpu or (not args.use_cpu and rank_id == 0):
        meta_new = merge_scores(gathered_list, dataset_generator.meta.copy(), columns)
        meta_new.to_csv(out_path, index=False)
        print(f"New meta with LPIPS motion scores saved to '{out_path}'.")

if __name__ == "__main__":
    main()