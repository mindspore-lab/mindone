import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))
from opensora.dataset.text_dataset import create_dataloader
from opensora.models.text_encoder.t5 import T5Embedder
from opensora.utils.ms_utils import init_env
from opensora.utils.utils import get_precision

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)

skip_vae = True


def read_captions_from_csv(path, caption_column="caption"):
    df = pd.read_csv(path, usecols=[caption_column])
    captions = df[caption_column].values.tolist()
    return captions


def read_captions_from_txt(path):
    captions = []
    with open(path, "r") as fp:
        for line in fp:
            captions.append(line.strip())
    return captions


def main(args):
    set_logger(name="", output_dir="logs/infer_t5")

    rank_id, device_num = init_env(
        mode=args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device_target,
        jit_level=args.jit_level,
    )
    print(f"rank_id {rank_id}, device_num {device_num}")

    # build dataloader for large amount of captions
    if args.data_file_path is not None:
        ds_config = dict(
            data_file_path=args.data_file_path,
            file_column=args.file_column,
            caption_column=args.caption_column,
            tokenizer=None,  # tokenizer,
        )
        dataset = create_dataloader(
            ds_config,
            args.batch_size,
            ds_name="text",
            num_parallel_workers=12,
            max_rowsize=32,
            shuffle=False,  # be in order
            device_num=device_num,
            rank_id=rank_id,
            drop_remainder=False,
        )
        dataset_size = dataset.get_dataset_size()
        logger.info(f"Num batches: {dataset_size}")

    logger.info("T5 init")
    text_encoder = T5Embedder(
        dir_or_name=args.text_encoder_name,
        model_max_length=args.model_max_length,
        cache_dir="./",
    )

    # mixed precision
    text_encoder_dtype = get_precision(args.precision)
    text_encoder = auto_mixed_precision(text_encoder, amp_level="O2", dtype=text_encoder_dtype)
    text_encoder.dtype = text_encoder_dtype
    logger.info(f"Use amp level O2 for text encoder T5 with dtype={text_encoder_dtype}")
    # infer
    if args.data_file_path is not None:
        if args.output_path is None:
            output_dir = os.path.dirname(args.data_file_path)
        else:
            output_dir = args.output_path
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output embeddings will be saved: {output_dir}")
        logger.info("Start embedding...")

        ds_iter = dataset.create_dict_iterator(1, output_numpy=True)
        for step, data in tqdm(enumerate(ds_iter), total=dataset_size):
            start_time = time.time()
            file_paths = data["file_path"]
            captions = data["caption"]
            captions = [str(captions[i]) for i in range(len(captions))]
            # print(captions)

            text_tokens, mask = text_encoder.get_text_tokens_and_mask(captions, return_tensor=True)
            text_emb = text_encoder(text_tokens, mask)

            end_time = time.time()
            time_cost = end_time - start_time

            # save the embeddings aligning to video frames
            for i in range(text_emb.shape[0]):
                fn = Path(str(file_paths[i])).with_suffix(".npz")
                npz_fp = os.path.join(output_dir, fn)
                if not os.path.exists(os.path.dirname(npz_fp)):
                    os.makedirs(os.path.dirname(npz_fp))

                np.savez(
                    npz_fp,
                    mask=mask[i].float().asnumpy().astype(np.uint8),
                    text_emb=text_emb[i].float().asnumpy().astype(np.float32),
                    # tokens=text_tokens[i].asnumpy(), #.astype(np.int32),
                )
        logger.info(f"Current step time cost: {time_cost:0.3f}s")
        logger.info(f"Done. Embeddings saved in {output_dir}")

    else:
        if args.output_path is None:
            output_dir = "samples/t5_embed"
        else:
            output_dir = args.output_path
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output embeddings will be saved: {output_dir}")

        # get captions from cfg or prompt_path
        if args.prompt_path is not None:
            if args.prompt_path.endswith(".csv"):
                captions = read_captions_from_csv(args.prompt_path, args.caption_column)
            elif args.prompt_path.endswith(".txt"):
                captions = read_captions_from_txt(args.prompt_path, args.caption_column)
        else:
            captions = args.captions
        logger.info(f"Number of captions: {len(captions)}")

        for i in tqdm(range(0, len(captions), args.batch_size)):
            batch_prompts = captions[i : i + args.batch_size]
            ns = len(batch_prompts)

            batch_text_tokens, batch_mask = text_encoder.get_text_tokens_and_mask(batch_prompts, return_tensor=True)
            batch_text_emb = text_encoder(batch_text_tokens, batch_mask)

            # save result
            batch_mask = batch_mask.asnumpy().astype(np.uint8)
            batch_text_emb = batch_text_emb.asnumpy().astype(np.float32)
            batch_text_tokens = batch_text_tokens.asnumpy()
            for j in range(ns):
                global_idx = i + j
                prompt = "-".join((batch_prompts[j].replace("/", "").split(" ")[:10]))
                save_fp = f"{output_dir}/{global_idx:03d}-{prompt}.npz"
                np.savez(
                    save_fp,
                    mask=batch_mask[j : j + 1],
                    text_emb=batch_text_emb[j : j + 1],
                    tokens=batch_text_tokens[j : j + 1],
                )

        logger.info(f"Finished. Embeddeings saved in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments. It can contain captions.",
    )
    parser.add_argument(
        "--data_file_path",
        default=None,
        type=str,
        help="path to csv annotation file",
    )
    parser.add_argument(
        "--prompt_path", default=None, type=str, help="path to a txt file, each line of which is a text prompt"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="output dir to save the embeddings, if None, will treat the parent dir of data_file_path as output dir.",
    )
    parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    # MS new args
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp32`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--amp_level",
        default="O2",
        type=str,
        help="mindspore amp level, O1: most fp32, only layers in whitelist compute in fp16 (dense, conv, etc); \
            O2: most fp16, only layers in blacklist compute in fp32 (batch norm etc)",
    )
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument(
        "--captions",
        type=str,
        nargs="+",
        help="A list of text captions to be generated with",
    )
    parser.add_argument(
        "--file_column",
        default="path",
        help="The column of file path in `data_file_path`. Defaults to `path`.",
    )
    parser.add_argument(
        "--caption_column",
        default="cap",
        help="The column of caption file path in `data_file_path`. Defaults to `cap`.",
    )

    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--model_max_length", type=int, default=300)
    parser.add_argument("--jit_level", default="O0", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    default_args = parser.parse_args()
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(__dir__, ".."))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            # _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(
                **dict(
                    captions=cfg["captions"],
                    text_encoder_name=cfg["text_encoder_name"],
                )
            )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
