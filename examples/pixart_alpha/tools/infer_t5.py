#!/usr/bin/env python
import argparse
import csv
import logging
import os
from typing import Tuple

import numpy as np
from modules.text_encoder import T5Embedder
from tqdm import tqdm

import mindspore as ms
from mindspore.dataset import GeneratorDataset

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


class TextDataset:
    def __init__(self, csv_path: str, path_column: str = "dir", caption_column: str = "caption") -> None:
        logger.info(f"loading annotations from {csv_path} ...")
        with open(csv_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))

        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")

        self.path_column = path_column
        self.caption_column = caption_column

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[str, str]:
        row = self.dataset[idx]
        path = row[self.path_column]
        caption = row[self.caption_column]
        return path, caption


def init_env(args) -> None:
    set_random_seed(args.seed)
    ms.set_context(mode=args.mode, device_target=args.device_target, jit_config=dict(jit_level="O2"))


def main(args):
    set_logger(name="", output_dir="logs/t5")

    # init env
    args = parse_args()
    init_env(args)

    # build dataloader
    dataset = TextDataset(args.csv_path, path_column=args.path_column, caption_column=args.caption_column)
    dataset = GeneratorDataset(dataset, column_names=["path", "text"], shuffle=False)
    dataset = dataset.batch(args.batch_size, drop_remainder=False)

    # model initiate and weight loading
    network = T5Embedder(
        args.t5_root,
        use_text_preprocessing=args.clean_caption,
        model_max_length=args.t5_max_length,
        pretrained_ckpt=os.path.join(args.t5_root, "model.ckpt"),
    )
    network.set_train(False)
    for param in network.trainable_params():
        param.requires_grad = False

    if args.dtype == "fp16":
        model_dtype = ms.float16
        network = auto_mixed_precision(network, amp_level="O2", dtype=model_dtype)
    elif args.dtype == "bf16":
        model_dtype = ms.bfloat16
        network = auto_mixed_precision(network, amp_level="O2", dtype=model_dtype)
    else:
        model_dtype = ms.float32

    if args.output_path is None:
        output_dir = os.path.dirname(args.csv_path)
    else:
        output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output embeddings will be saved: {output_dir}")
    logger.info("Start embedding...")

    ds_iter = dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)
    for paths, texts in tqdm(ds_iter, total=len(dataset)):
        paths, texts = paths.tolist(), texts.tolist()
        text_tokens, mask = network.get_text_tokens_and_mask(texts, return_tensor=True)
        text_emb = network(text_tokens, mask).asnumpy()
        mask = mask.asnumpy()
        assert text_emb.shape[0] == len(paths)

        # save the embeddings aligning to video frames
        for i in range(text_emb.shape[0]):
            filename = os.path.splitext(paths[i])[0] + ".npz"
            filepath = os.path.join(output_dir, filename)
            np.savez_compressed(filepath, mask=mask[i], text_emb=text_emb[i])
    logger.info(f"Done. Embeddings saved in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract t5 feature from a csv file")
    parser.add_argument("--csv_path", required=True, help="path to csv annotation file.")
    parser.add_argument("--path_column", default="dir", help="column name of image path in csv file.")
    parser.add_argument("--caption_column", default="text", help="column name of caption in csv file.")
    parser.add_argument(
        "--output_path",
        help="output dir to save the embeddings, if None, will treat the parent dir of csv_path as output dir.",
    )
    parser.add_argument(
        "--t5_root", default="models/t5-v1_1-xxl", help="Path storing the T5 checkpoint and tokenizer configure file."
    )
    parser.add_argument("--t5_max_length", type=int, default=120, help="T5's embedded sequence length.")
    # MS new args
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp32`, which corresponds to ms.float16",
    )
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
