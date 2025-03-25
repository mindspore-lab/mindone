import argparse
import os

import numpy as np
import pandas as pd
from pipeline.datasets.utils import extract_frames
from pipeline.scoring.lpips.lpips import LPIPS
from pipeline.scoring.utils import merge_scores
from tqdm import tqdm

import mindspore as ms
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.mint.distributed import all_gather, get_rank, get_world_size, init_process_group


class VideoTextDataset:
    def __init__(self, meta_path, seconds=1, target_size=(224, 224)):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.seconds = seconds
        self.target_size = target_size  # target reshape size (downsample for images)

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample["path"]

        images, timestamps = extract_frames(path, seconds=self.seconds, backend="decord", return_timestamps=True)
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
        "--seconds", type=int, default=1, help="Interval (in seconds) at which frames are sampled from the video."
    )
    parser.add_argument("--target_height", type=int, default=224, help="Target image height to be processed")
    parser.add_argument("--target_width", type=int, default=224, help="Target image width to be processed")
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
        init_process_group()
        rank_id = get_rank()
        rank_size = get_world_size()

    model = LPIPS()
    model.load_from_pretrained(args.lpips_ckpt_path)
    model.set_train(False)

    dataset_generator = VideoTextDataset(
        meta_path=args.meta_path, seconds=args.seconds, target_size=(args.target_height, args.target_width)
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

    # compute optical flow scores
    indices_list = []
    scores_list = []
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
            else:
                # compute time differences between frames
                delta_times = times[1:] - times[:-1]  # Shape: [N-1]

                # prepare pairs of frames
                imgs_0 = imgs[:-1]  # [N-1, C, H, W]
                imgs_1 = imgs[1:]  # [N-1, C, H, W]

                scores = model(imgs_0, imgs_1)  # [N-1, 1]
                scores = scores.squeeze().asnumpy()  # [N-1]

                # weighted sum of scores
                weighted_scores = scores * delta_times

                total_time = times[-1] - times[0]
                avg_score = weighted_scores.sum() / total_time

            indices_list.append(index)
            scores_list.append(avg_score)

    # Allgather results if necessary
    if not args.use_cpu:
        indices_list = Tensor(indices_list, dtype=ms.int64)
        scores_list = Tensor(scores_list, dtype=ms.float32)

        indices_list_all = [Tensor(np.zeros(indices_list.shape, dtype=np.int64)) for _ in range(rank_size)]
        scores_list_all = [Tensor(np.zeros(scores_list.shape, dtype=np.float32)) for _ in range(rank_size)]

        all_gather(indices_list_all, indices_list)
        all_gather(scores_list_all, scores_list)

        concat = ops.Concat(axis=0)
        indices_list_all = concat(indices_list_all).asnumpy().tolist()
        scores_list_all = concat(scores_list_all).asnumpy().tolist()
    else:
        indices_list_all = indices_list
        scores_list_all = scores_list

    if args.use_cpu or (not args.use_cpu and rank_id == 0):
        meta_new = merge_scores([(indices_list_all, scores_list_all)], dataset_generator.meta, column="lpips")
        meta_new.to_csv(out_path, index=False)
        print(f"New meta with LPIPS motion scores saved to '{out_path}'.")


if __name__ == "__main__":
    main()
