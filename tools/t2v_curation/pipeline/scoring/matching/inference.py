import argparse
import os

import mindspore as ms
import mindspore.dataset as ds
from mindspore import context, Tensor, ops
from mindspore.communication import get_rank, get_group_size, init

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline.datasets.utils import extract_frames, is_video, pil_loader
from pipeline.scoring.clip import CLIPImageProcessor, CLIPModel, parse, CLIPTokenizer
from pipeline.scoring.utils import merge_scores, NUM_FRAMES_POINTS

class VideoTextDataset:
    def __init__(self, meta_path, transform=None, num_frames=3):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.transform = transform
        self.points = NUM_FRAMES_POINTS[num_frames]

    def __getitem__(self, index):
        """
        Args:
            text: text to be evaluated for matching, this could be from
            - the csv file: after captioning, in this case the input is None
            - predefined list in .txt or .csv files
            - command line input: support ***1*** text string only
        Returns:
        """
        sample = self.meta.iloc[index]
        path = sample['path']

        # extract frames
        if not is_video(path):
            images = [pil_loader(path)]
        else:
            num_frames = sample["num_frames"] if "num_frames" in sample else None
            images = extract_frames(path, points=self.points, backend="decord", num_frames=num_frames)

        # transform & stack
        if self.transform is not None:
            images = [self.transform(img) for img in images]
        images = np.stack([img.asnumpy() for img in images])

         # read from csv directly if exists, else return None
        text = sample.get('text', '')
        return images, text, index

    def __len__(self):
        return len(self.meta)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")

    parser.add_argument("--use_cpu", action="store_true", help="Whether to use CPU")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--num_frames", type=int, default=1, help="Number of frames to extract, support 1, 2, and 3.")
    parser.add_argument("--option", type=str, default=None, help="Option filtering string")

    parser.add_argument("--skip_if_existing", action="store_true")

    parser.add_argument("--config", type=str, default="pipeline/scoring/clip/configs/clip_vit_l_14.yaml", help="YAML config files for ms backend.")
    parser.add_argument("--ckpt_path_clip", type=str, default="pretrained_models/clip_vit_l_14.ckpt", help = "load clip model checkpoint.")
    parser.add_argument("--tokenizer_path", type=str, default="pretrained_models/bpe_simple_vocab_16e6.txt.gz", help="load clip tokenizer checkpoint.")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    if args.option is not None:
        out_path = f"{wo_ext}_{args.option}{ext}"
    else:
        out_path = f"{wo_ext}_matching{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    # graph mode with Ascend
    if not args.use_cpu:
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        init()

    # clip model
    config = parse(args.config, args.ckpt_path_clip)
    model = CLIPModel(config)
    processor = CLIPImageProcessor()
    tokenizer = CLIPTokenizer(args.tokenizer_path)
    logit_scale = np.exp(model.logit_scale.asnumpy())

    # build dataset
    raw_dataset = VideoTextDataset(args.meta_path, transform=processor, num_frames=args.num_frames)
    if not args.use_cpu:
        rank_id = get_rank()
        rank_size = get_group_size()
        dataset = ds.GeneratorDataset(source=raw_dataset, column_names=['image', 'text', 'index'], shuffle=False,
                                      num_shards=rank_size, shard_id=rank_id)
    else:
        dataset = ds.GeneratorDataset(source=raw_dataset, column_names=['image', 'text', 'index'], shuffle=False)
    dataset = dataset.batch(args.bs, drop_remainder=False)
    iterator = dataset.create_dict_iterator(num_epochs = 1, output_numpy=True)

    # compute matching scores
    indices_list = []
    scores_list = []
    model.set_train(False)
    for batch in tqdm(iterator):
        images = Tensor(batch["image"])
        text = [t.item() for t in batch["text"]]
        if text[0] == '' and args.option is not None:
            text = args.option
        indices = batch["index"]
        B, N, _, C, H, W = images.shape # (batch_size, num_frames, 1, channels, height, width)
        reshape = ops.Reshape()
        images = reshape(images, (B * N, C, H, W))
        image_features = model.get_image_features(images)
        image_features = reshape(image_features, (B, N, -1))
        # text preprocess
        # add [] to match dimension (support when using a single string instead of a list of strings)
        if isinstance(text, str):
            text = Tensor(B * [tokenizer(text, padding = 'max_length', max_length = 77, truncation = True)['input_ids']])
        else:
            text = Tensor(tokenizer(text, padding = 'max_length', max_length = 77, truncation = True)['input_ids'])
        text_features = model.get_text_features(text)

        normalize = ops.L2Normalize(axis = -1)
        image_features = normalize(image_features)
        text_features = normalize(text_features)
        text_features = reshape(text_features, (B, 1, -1))

        # TODO: may provide options with max or average - I think both have merits (here we implement max)
        # report the max similarity scores if given multiple frames
        sum = ops.ReduceSum()
        clip_scores = logit_scale * sum(image_features * text_features, axis = -1).asnumpy() # (B, N)
        max_clip_scores = np.max(clip_scores, axis=-1)  # B

        indices_list.extend(indices.tolist())
        scores_list.extend(max_clip_scores.tolist())

    if not args.use_cpu:
        allgather = ops.AllGather()
        indices_list = Tensor(indices_list, dtype=ms.int64)
        scores_list = Tensor(scores_list, dtype=ms.float32)
        indices_list = allgather(indices_list).asnumpy().tolist()
        scores_list = allgather(scores_list).asnumpy().tolist()

    if args.use_cpu or (not args.use_cpu and rank_id == 0):
        meta_local = merge_scores([(indices_list, scores_list)], raw_dataset.meta, column="match")
        meta_local.to_csv(out_path, index = False)
        print(meta_local)
        print(f"New meta with matching scores saved to '{out_path}'.")



if __name__ == "__main__":
    main()
