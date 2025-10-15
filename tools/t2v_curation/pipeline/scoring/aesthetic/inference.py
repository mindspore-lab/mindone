import argparse
import os
import sys

import numpy as np
import pandas as pd
from pipeline.datasets.utils import extract_frames, is_video, pil_loader
from pipeline.scoring.utils import NUM_FRAMES_POINTS, merge_scores
from tqdm import tqdm
from transformers import AutoProcessor

import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, load_checkpoint, load_param_into_net
from mindspore.mint.distributed import all_gather, get_rank, get_world_size, init_process_group

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../../.."))
sys.path.insert(0, mindone_lib_path)

from mindone.transformers import CLIPModel  # noqa: E402


class VideoTextDataset:
    def __init__(self, meta_path, transform=None, num_frames=3):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.transform = transform
        self.points = NUM_FRAMES_POINTS[num_frames]

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample["path"]

        # extract frames
        if not is_video(path):
            images = [pil_loader(path)]
        else:
            num_frames = sample["num_frames"] if "num_frames" in sample else None
            images = extract_frames(path, points=self.points, backend="decord", num_frames=num_frames)

        # transform & stack
        if self.transform is not None:
            images = [self.transform(images=img, return_tensors="np").pixel_values for img in images]
        images = np.stack(images)

        return index, images

    def __len__(self):
        return len(self.meta)


class MLP(nn.Cell):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.SequentialCell(
            [
                nn.Dense(self.input_size, 1024),
                nn.Dropout(p=0.2),
                nn.Dense(1024, 128),
                nn.Dropout(p=0.2),
                nn.Dense(128, 64),
                nn.Dropout(p=0.1),
                nn.Dense(64, 16),
                nn.Dense(16, 1),
            ]
        )

    def construct(self, x):
        return self.layers(x)


class AestheticScorer(nn.Cell):
    def __init__(self, input_size=768):
        super().__init__()
        self.mlp = MLP(input_size)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def construct(self, x):
        image_features = self.clip.get_image_features(x)
        normalize = ops.L2Normalize(axis=-1)
        image_features = normalize(image_features).astype(ms.float32)
        return self.mlp(image_features)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")

    parser.add_argument("--use_cpu", action="store_true", help="Whether to use CPU")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--num_frames", type=int, default=1, help="Number of frames to extract, support 1, 2, and 3.")

    parser.add_argument("--skip_if_existing", action="store_true")

    parser.add_argument(
        "--ckpt_path_aes", type=str, default="pretrained_models/aesthetic.ckpt", help="load aesthetic model checkpoint."
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
    out_path = f"{wo_ext}_aes{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    # graph mode with Ascend
    if not args.use_cpu:
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        init_process_group()

    model = AestheticScorer()
    preprocess = model.processor
    param_dict = load_checkpoint(args.ckpt_path_aes)
    load_param_into_net(model.mlp, param_dict)

    raw_dataset = VideoTextDataset(args.meta_path, transform=preprocess, num_frames=args.num_frames)
    if not args.use_cpu:
        rank_id = get_rank()
        rank_size = get_world_size()
        dataset = ds.GeneratorDataset(
            source=raw_dataset, column_names=["index", "images"], shuffle=False, num_shards=rank_size, shard_id=rank_id
        )
    else:
        dataset = ds.GeneratorDataset(source=raw_dataset, column_names=["index", "images"], shuffle=False)
    dataset = dataset.batch(args.bs, drop_remainder=False)
    iterator = dataset.create_dict_iterator(num_epochs=1)

    # compute aesthetic scores
    indices_list = []
    scores_list = []
    model.set_train(False)
    for batch in tqdm(iterator):
        indices = batch["index"]
        images = batch["images"]

        B, N, _, C, H, W = images.shape  # (batch_size, num_frames, 1, channels, height, width)
        reshape = ops.Reshape()
        images = reshape(images, (B * N, C, H, W))
        scores = model(images)
        scores = reshape(scores, (B, N))
        scores = scores.mean(axis=-1)
        scores_np = scores.asnumpy()

        indices_list.extend(indices.tolist())
        scores_list.extend(scores_np.tolist())

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
        meta_local = merge_scores([(indices_list_all, scores_list_all)], raw_dataset.meta, column="aes")
        meta_local.to_csv(out_path, index=False)
        print(meta_local)
        print(f"New meta with aesthetic scores saved to '{out_path}'.")


if __name__ == "__main__":
    main()
