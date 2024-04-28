import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from opensora.data.text_dataset import create_dataloader
from opensora.models.text_encoders import get_text_encoder_and_tokenizer
from opensora.utils.model_utils import str2bool  # _check_cfgs_in_parser
from tqdm import tqdm

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed
from mindone.utils.misc import to_abspath

logger = logging.getLogger(__name__)

skip_vae = True


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    rank_id = 0
    ms.set_context(
        mode=args.mode,
        device_target=args.device_target,
        device_id=device_id,
    )
    if args.precision_mode is not None:
        ms.set_context(ascend_config={"precision_mode": args.precision_mode})

    device_num = 1
    return device_id, rank_id, device_num


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    device_id, rank_id, device_num = init_env(args)
    print(f"D--: rank_id {rank_id}, device_num {device_num}")
    set_random_seed(args.seed)

    # build dataloader for large amount of captions
    if args.csv_path is not None:
        ds_config = dict(
            csv_path=to_abspath(__dir__, args.csv_path),
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
    ckpt_path = to_abspath(__dir__, args.t5_model_dir)
    text_encoder, tokenizer = get_text_encoder_and_tokenizer("t5", ckpt_path)
    text_encoder.set_train(False)
    for param in text_encoder.get_parameters():  # freeze latte_model
        param.requires_grad = False

    logger.info("Start embedding...")

    # infer
    if args.csv_path is not None:
        ds_iter = dataset.create_dict_iterator(1, output_numpy=True)
        if args.output_dir is None:
            output_folder = os.path.dirname(args.csv_path)
        else:
            output_folder = args.output_dir
        os.makedirs(output_folder, exist_ok=True)

        logger.info(f"Output embeddings will be saved: {output_folder}")

        for step, data in tqdm(enumerate(ds_iter)):
            start_time = time.time()

            file_paths = data["file_path"]
            captions = data["caption"]
            captions = [str(captions[i]) for i in range(len(captions))]
            # print(captions)

            text_tokens, mask = text_encoder.get_text_tokens_and_mask(captions, return_tensor=True)
            text_emb = text_encoder(text_tokens, mask)

            end_time = time.time()
            logger.info(f"Time cost: {end_time-start_time:0.3f}s")

            # save the embeddings aligning to video frames
            for i in range(text_emb.shape[0]):
                fn = Path(str(file_paths[i])).with_suffix(".npz")
                npz_fp = os.path.join(output_folder, fn)
                if not os.path.exists(os.path.dirname(npz_fp)):
                    os.makedirs(os.path.dirname(npz_fp))

                np.savez(
                    npz_fp,
                    mask=mask[i].asnumpy().astype(np.uint8),
                    text_emb=text_emb[i].asnumpy().astype(np.float32),
                    # tokens=text_tokens[i].asnumpy(), #.astype(np.int32),
                )

        logger.info(f"Done. Embeddings saved in {output_folder}")

    else:
        text_tokens = []
        mask = []
        text_emb = []
        for i in range(0, len(args.captions), args.batch_size):
            batch_text_tokens, batch_mask = text_encoder.get_text_tokens_and_mask(
                args.captions[i : i + args.batch_size], return_tensor=True
            )
            logger.info(f"Num tokens: {batch_mask.asnumpy().sum(1)}")
            batch_text_emb = text_encoder(batch_text_tokens, batch_mask)

            text_tokens.append(batch_text_tokens.asnumpy())
            mask.append(batch_mask.asnumpy().astype(np.uint8))
            text_emb.append(batch_text_emb.asnumpy())
        text_tokens = np.concatenate(text_tokens)
        mask = np.concatenate(mask)
        text_emb = np.concatenate(text_emb)
        np.savez(args.output_path, tokens=text_tokens, mask=mask, text_emb=text_emb)
        print("Embeddeings saved in ", args.output_path)


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
        help="path to csv annotation file, If None, video_caption.csv is expected to live under `data_path`",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output dir to save the embeddings, if None, will treat the parent dir of csv_path as output dir.",
    )
    parser.add_argument("--caption_column", type=str, default="caption", help="caption column num in csv")
    parser.add_argument("--t5_model_dir", default="models/t5-v1_1-xxl", type=str, help="the T5 cache folder path")
    # MS new args
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
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
    parser.add_argument("--output_path", type=str, default="outputs/t5_embed.npz", help="path to save t5 embedding")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")

    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))

    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
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
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
