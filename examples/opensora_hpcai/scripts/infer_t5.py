import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import mindspore as ms
from mindspore.communication.management import get_group_size, get_rank, init

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from opensora.datasets.text_dataset import create_dataloader
from opensora.models.text_encoder.t5 import get_text_encoder_and_tokenizer
from opensora.utils.cond_data import read_captions_from_csv, read_captions_from_txt
from opensora.utils.model_utils import str2bool  # _check_cfgs_in_parser

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.misc import to_abspath
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)

skip_vae = True


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    jit_level: str = "O0",
):
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
        ms.reset_auto_parallel_context()

        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )

        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        logger.info(dict(zip(var_info, var_value)))

    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )

    try:
        if jit_level in ["O0", "O1", "O2"]:
            ms.set_context(jit_config={"jit_level": jit_level})
        else:
            logger.warning(
                f"Unsupport jit_level: {jit_level}. The framework automatically selects the execution method"
            )
    except Exception:
        logger.warning(
            "The current jit_level is not suitable because current MindSpore version or mode does not match,"
            "please ensure the MindSpore version >= ms2.3_0615, and use GRAPH_MODE."
        )

    return rank_id, device_num


def main(args):
    set_logger(name="", output_dir="logs/infer_t5")

    rank_id, device_num = init_env(args.mode, args.seed, args.use_parallel, device_target=args.device_target)
    print(f"rank_id {rank_id}, device_num {device_num}")

    # build dataloader for large amount of captions
    if args.csv_path is not None:
        ds_config = dict(
            csv_path=args.csv_path,
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

    # model initiate and weight loading
    ckpt_path = args.t5_model_dir
    text_encoder, tokenizer = get_text_encoder_and_tokenizer("t5", ckpt_path, model_max_length=args.model_max_length)
    text_encoder.set_train(False)
    for param in text_encoder.get_parameters():  # freeze latte_model
        param.requires_grad = False

    dtype_map = {"fp16": ms.float16, "bf16": ms.bfloat16}
    if args.dtype in ["fp16", "bf16"]:
        text_encoder = auto_mixed_precision(text_encoder, amp_level=args.amp_level, dtype=dtype_map[args.dtype])

    # infer
    if args.csv_path is not None:
        if args.output_path is None:
            output_dir = os.path.dirname(args.csv_path)
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
                    mask=mask[i].asnumpy().astype(np.uint8),
                    text_emb=text_emb[i].asnumpy().astype(np.float32),
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
                captions = read_captions_from_csv(args.prompt_path)
            elif args.prompt_path.endswith(".txt"):
                captions = read_captions_from_txt(args.prompt_path)
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
        "--csv_path",
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
        help="output dir to save the embeddings, if None, will treat the parent dir of csv_path as output dir.",
    )
    parser.add_argument("--caption_column", type=str, default="caption", help="caption column num in csv")
    parser.add_argument("--t5_model_dir", default="models/t5-v1_1-xxl", type=str, help="the T5 cache folder path")
    parser.add_argument("--model_max_length", type=int, default=120, help="T5's embedded sequence length.")
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
        "--dtype",
        default="fp32",
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
        "--use_recompute",
        default=False,
        type=str2bool,
        help="whether use recompute.",
    )
    parser.add_argument(
        "--captions",
        type=str,
        nargs="+",
        help="A list of text captions to be generated with",
    )
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")

    default_args = parser.parse_args()
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(__dir__, ".."))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = to_abspath(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            # _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(
                **dict(
                    captions=cfg["captions"],
                    t5_model_dir=cfg["t5_model_dir"],
                )
            )
    args = parser.parse_args()
    # convert to absolute path, necessary for modelarts
    args.csv_path = to_abspath(abs_path, args.csv_path)
    args.prompt_path = to_abspath(abs_path, args.prompt_path)
    args.output_path = to_abspath(abs_path, args.output_path)
    args.t5_model_dir = to_abspath(abs_path, args.t5_model_dir)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
