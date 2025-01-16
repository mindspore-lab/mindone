import argparse
import os

import numpy as np
import pandas as pd
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size, init
from mindspore import Tensor, load_checkpoint, save_checkpoint, load_param_into_net
from tqdm import tqdm

from pipeline.datasets.utils import extract_frames, pil_loader, is_video
from pipeline.scoring.clip import CLIPImageProcessor, CLIPModel, parse
from pipeline.scoring.utils import merge_scores, NUM_FRAMES_POINTS


class VideoTextDataset:
    def __init__(self, meta_path, transform = None, num_frames = 3):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.transform = transform
        self.points = NUM_FRAMES_POINTS[num_frames]

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample['path']

        # extract frames
        if not is_video(path):
            images = [pil_loader(path)]
        else:
            num_frames = sample["num_frames"] if "num_frames" in sample else None
            images = extract_frames(path, points = self.points, backend = "decord", num_frames = num_frames)

        # transform & stack
        if self.transform is not None:
            images = [self.transform(img) for img in images]
        images = np.stack([img.asnumpy() for img in images])

        return index, images

    def __len__(self):
        return len(self.meta)

class MLP(nn.Cell):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.SequentialCell([
            nn.Dense(self.input_size, 1024),
            nn.Dropout(p = 0.2),
            nn.Dense(1024, 128),
            nn.Dropout(p = 0.2),
            nn.Dense(128, 64),
            nn.Dropout(p = 0.1),
            nn.Dense(64, 16),
            nn.Dense(16, 1),
        ])

    def construct(self, x):
        return self.layers(x)

class AestheticScorer(nn.Cell):
    def __init__(self, input_size = 768, config = 'pipeline/scoring/clip/configs/clip_vit_l_14.yaml', ckpt_path = None):
        super().__init__()
        self.mlp = MLP(input_size)
        config = parse(config, ckpt_path)
        self.clip = CLIPModel(config)
        self.processor = CLIPImageProcessor()

    def construct(self, x):
        image_features = self.clip.get_image_features(x)
        normalize = ops.L2Normalize(axis = -1)
        image_features = normalize(image_features).astype(ms.float32)
        return self.mlp(image_features)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")

    parser.add_argument("--use_cpu", action="store_true", help="Whether to use CPU")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--num_frames", type=int, default=1, help="Number of frames to extract, support 1, 2, and 3.")

    parser.add_argument("--skip_if_existing", action="store_true")

    parser.add_argument("--config", type=str, default="pipeline/scoring/clip/configs/clip_vit_l_14.yaml", help="YAML config files for ms backend.")
    parser.add_argument("--ckpt_path_clip", type=str, default="pretrained_models/clip_vit_l_14.ckpt", help = "load clip model checkpoint.")
    parser.add_argument("--ckpt_path_aes", type=str, default="pretrained_models/aesthetic.ckpt",
                        help="load aesthetic model checkpoint.")

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
        ms.set_auto_parallel_context(parallel_mode = ms.ParallelMode.DATA_PARALLEL)
        init()

    model = AestheticScorer(config = args.config, ckpt_path = args.ckpt_path_clip)
    preprocess = model.processor.preprocess
    param_dict = load_checkpoint(args.ckpt_path_aes)
    load_param_into_net(model.mlp, param_dict)

    raw_dataset = VideoTextDataset(args.meta_path, transform=preprocess, num_frames=args.num_frames)
    if not args.use_cpu:
        rank_id = get_rank()
        rank_size = get_group_size()
        dataset = ds.GeneratorDataset(source=raw_dataset, column_names=['index', 'images'], shuffle=False,
                                      num_shards = rank_size, shard_id = rank_id)
    else:
        dataset = ds.GeneratorDataset(source = raw_dataset, column_names = ['index', 'images'], shuffle = False)
    dataset = dataset.batch(args.bs, drop_remainder=False)
    iterator = dataset.create_dict_iterator(num_epochs = 1)

    # compute aesthetic scores
    indices_list = []
    scores_list = []
    model.set_train(False)
    for batch in tqdm(iterator):
        indices = batch["index"]
        images = batch["images"]

        B, N, _, C, H, W = images.shape # (batch_size, num_frames, 1, channels, height, width)
        reshape = ops.Reshape()
        images = reshape(images, (B * N, C, H, W))
        scores = model(images)
        scores = reshape(scores, (B, N))
        scores = scores.mean(axis=-1)
        scores_np = scores.asnumpy()

        indices_list.extend(indices.tolist())
        scores_list.extend(scores_np.tolist())

    if not args.use_cpu:
        allgather = ops.AllGather()
        indices_list = Tensor(indices_list, dtype=ms.int64)
        scores_list = Tensor(scores_list, dtype=ms.float32)
        indices_list = allgather(indices_list).asnumpy().tolist()
        scores_list = allgather(scores_list).asnumpy().tolist()

    if args.use_cpu or (not args.use_cpu and rank_id == 0):
        meta_local = merge_scores([(indices_list, scores_list)], raw_dataset.meta, column="aes")
        meta_local.to_csv(out_path, index = False)
        print(meta_local)
        print(f"New meta with aesthetic scores saved to '{out_path}'.")

if __name__ == "__main__":
    main()
