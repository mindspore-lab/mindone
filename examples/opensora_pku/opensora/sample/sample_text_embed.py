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

import mindspore as ms

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))
from opensora.dataset.text_dataset import create_dataloader
from opensora.dataset.transform import t5_text_preprocessing as text_preprocessing
from opensora.npu_config import npu_config
from opensora.utils.message_utils import print_banner
from opensora.utils.utils import get_precision
from transformers import AutoTokenizer

from mindone.transformers import MT5EncoderModel
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
    set_logger(name="", output_dir="logs/infer_mt5")
    rank_id, device_num = npu_config.set_npu_env(args)
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

    print_banner("text encoder init")
    text_encoder_dtype = get_precision(args.text_encoder_precision)
    text_encoder, loading_info = MT5EncoderModel.from_pretrained(
        args.text_encoder_name,
        cache_dir=args.cache_dir,
        output_loading_info=True,
        mindspore_dtype=text_encoder_dtype,
        use_safetensors=True,
    )
    loading_info.pop("unexpected_keys")  # decoder weights are ignored
    logger.info(loading_info)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)

    # infer
    print_banner("Text prompts loading")
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
            captions = [
                text_preprocessing(
                    prompt,
                )
                for prompt in captions
            ]
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=args.model_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors=None,
            )
            text_tokens = ms.Tensor(text_inputs.input_ids)
            mask = ms.Tensor(text_inputs.attention_mask)

            text_emb = text_encoder(text_tokens, attention_mask=mask)
            text_emb = text_emb[0] if isinstance(text_emb, (list, tuple)) else text_emb

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
            output_dir = f"samples/{args.text_encoder_name}_embed"
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
            batch_prompts = [
                text_preprocessing(
                    prompt,
                )
                for prompt in batch_prompts
            ]
            text_inputs = tokenizer(
                batch_prompts,
                padding="max_length",
                max_length=args.model_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors=None,
            )
            batch_text_tokens = ms.Tensor(text_inputs.input_ids)
            batch_mask = ms.Tensor(text_inputs.attention_mask)

            batch_text_emb = text_encoder(batch_text_tokens, attention_mask=batch_mask)
            batch_text_emb = batch_text_emb[0] if isinstance(batch_text_emb, (list, tuple)) else batch_text_emb
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
    parser.add_argument("--text_encoder_name", type=str, default="google/mt5-xxl")
    parser.add_argument(
        "--cache_dir",
        default="./",
        type=str,
        help="The cache directory to the text encoder and tokenizer",
    )
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
        "--text_encoder_precision",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for text encoder. Default is `bf16`, which corresponds to ms.bfloat16",
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
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--jit_level", default="O0", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    parser.add_argument(
        "--jit_syntax_level", default="strict", choices=["strict", "lax"], help="Set jit syntax level: strict or lax"
    )
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
