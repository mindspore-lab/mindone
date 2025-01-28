import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from constants import PRECISIONS, PROMPT_TEMPLATE
from dataset.text_dataset import create_dataloader
from dataset.transform import text_preprocessing
from text_encoder import TextEncoder
from utils.message_utils import print_banner
from utils.ms_utils import init_env

from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo text encoders")
    parser.add_argument(
        "--text_encoder_choices",
        type=str,
        nargs="+",
        default=["llm", "clipL"],
        choices=["llm", "clipL"],
        help="Specify the text encode type.",
    )

    # text encoder llm
    parser.add_argument(
        "--text_encoder_name",
        type=str,
        default="llm",
        help="Name of the text encoder model.",
    )
    parser.add_argument(
        "--text_encoder_precision",
        type=str,
        default="bf16",
        choices=PRECISIONS,
        help="Precision mode for the text encoder model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="ckpts/text_encoder",
        help="File path of the ckpt of the text encoder.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="llm",
        help="Name of the tokenizer model.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="File path of the ckpt of the tokenizer.",
    )
    parser.add_argument(
        "--text_len",
        type=int,
        default=256,
        help="Maximum length of the text input.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="dit-llm-encode",
        choices=PROMPT_TEMPLATE,
        help="Image prompt template for the decoder-only text encoder model.",
    )
    parser.add_argument(
        "--prompt_template_video",
        type=str,
        default="dit-llm-encode-video",
        choices=PROMPT_TEMPLATE,
        help="Video prompt template for the decoder-only text encoder model.",
    )
    parser.add_argument(
        "--hidden_state_skip_layer",
        type=int,
        default=2,
        help="Skip layer for hidden states.",
    )
    parser.add_argument(
        "--apply_final_norm",
        action="store_true",
        help="Apply final normalization to the used text encoder hidden states.",
    )

    # text encoder clipL
    parser.add_argument(
        "--text_encoder_name_2",
        type=str,
        default="clipL",
        help="Name of the second text encoder model.",
    )
    parser.add_argument(
        "--text_encoder_precision_2",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the second text encoder model.",
    )
    parser.add_argument(
        "--text_encoder_path_2",
        type=str,
        default="ckpts/text_encoder_2",
        help="File path of the ckpt of the second text encoder.",
    )
    parser.add_argument(
        "--tokenizer_2",
        type=str,
        default="clipL",
        help="Name of the second tokenizer model.",
    )
    parser.add_argument(
        "--tokenizer_path_2",
        type=str,
        default=None,
        help="File path of the ckpt of the second tokenizer.",
    )
    parser.add_argument(
        "--text_len_2",
        type=int,
        default=77,
        help="Maximum length of the second text input.",
    )

    # mindspore settings
    parser.add_argument(
        "--mode",
        default=1,
        type=int,
        help="Specify the MindSpore mode: 0 for graph mode, 1 for pynative mode",
    )
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports ['O0', 'O1', 'O2']."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument(
        "--use_parallel",
        default=False,
        type=str2bool,
        help="use parallel",
    )
    parser.add_argument(
        "--device_target",
        type=str,
        default="Ascend",
        help="Ascend or GPU",
    )
    parser.add_argument(
        "--jit_syntax_level",
        default="strict",
        choices=["strict", "lax"],
        help="Set jit syntax level: strict or lax",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="batch size",
    )

    # others
    parser.add_argument(
        "--data_file_path",
        type=str,
        default=None,
        help="File path of prompts, must be a txt or csv file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="text prompt",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output dir to save the embeddings, if None, will treat the parent dir of data_file_path as output dir.",
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
    args = parser.parse_args()

    return args


def save_emb(output, output_2, output_dir, file_paths):
    num = output.hidden_state.shape[0] if output is not None else output_2.hidden_state.shape[0]
    for i in range(num):
        fn = Path(str(file_paths[i])).with_suffix(".npz")
        npz_fp = Path(output_dir) / fn
        if not os.path.exists(npz_fp.parent):
            os.makedirs(npz_fp.parent)
        np.savez(
            npz_fp,
            prompt_embeds=output.hidden_state[i].float().asnumpy().astype(np.float32) if output is not None else None,
            prompt_mask=output.attention_mask[i].float().asnumpy().astype(np.uint8) if output is not None else None,
            prompt_embeds_2=output_2.hidden_state[i].float().asnumpy().astype(np.float32)
            if output_2 is not None
            else None,
        )


def build_model(args, logger):
    prompt_template = PROMPT_TEMPLATE[args.prompt_template] if args.prompt_template is not None else None
    prompt_template_video = (
        PROMPT_TEMPLATE[args.prompt_template_video] if args.prompt_template_video is not None else None
    )
    if args.prompt_template_video is not None:
        crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
    elif args.prompt_template is not None:
        crop_start = PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
    else:
        crop_start = 0
    max_length = args.text_len + crop_start

    text_encoder, text_encoder_2 = None, None

    # llm
    if args.text_encoder_name in args.text_encoder_choices:
        text_encoder = TextEncoder(
            text_encoder_type=args.text_encoder_name,
            max_length=max_length,
            text_encoder_precision=args.text_encoder_precision,
            text_encoder_path=args.text_encoder_path,
            tokenizer_type=args.tokenizer,
            tokenizer_path=args.tokenizer_path if args.tokenizer_path is not None else args.text_encoder_path,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            apply_final_norm=args.apply_final_norm,
            logger=logger,
        )

    # clipL
    if args.text_encoder_name_2 in args.text_encoder_choices:
        text_encoder_2 = TextEncoder(
            text_encoder_type=args.text_encoder_name_2,
            max_length=args.text_len_2,
            text_encoder_precision=args.text_encoder_precision_2,
            text_encoder_path=args.text_encoder_path_2,
            tokenizer_type=args.tokenizer_2,
            tokenizer_path=args.tokenizer_path_2 if args.tokenizer_path_2 is not None else args.text_encoder_path_2,
            logger=logger,
        )

    return text_encoder, text_encoder_2


def main(args):
    set_logger(name="", output_dir="logs/text_embed")
    rank_id, device_num = init_env(
        mode=args.mode,
        distributed=args.use_parallel,
        device_target=args.device_target,
        jit_level=args.jit_level,
        jit_syntax_level=args.jit_syntax_level,
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
            num_parallel_workers=1,
            max_rowsize=32,
            shuffle=False,  # be in order
            device_num=device_num,
            rank_id=rank_id,
            drop_remainder=False,
        )
        dataset_size = dataset.get_dataset_size()
        logger.info(f"Num batches: {dataset_size}")
    elif args.prompt is not None:
        data = {}
        prompt_fn = "-".join((args.prompt.replace("/", "").split(" ")[:16]))
        data["file_path"] = ["./{}.npz".format(prompt_fn)]
        data["caption"] = [args.prompt]
        dataset = None
        dataset_size = 1
        prompt_iter = iter([data])
    else:
        raise ValueError("Either data_file_path or prompt has to be provided")

    print_banner("text encoder init")
    text_encoder, text_encoder_2 = build_model(args, logger)

    # infer
    print_banner("Text prompts loading")
    if args.output_path is None:
        output_dir = Path(args.data_file_path).parent if args.data_file_path is not None else "./"
    else:
        output_dir = Path(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output embeddings will be saved: {output_dir}")
    logger.info("Start embedding...")
    if dataset is not None:
        ds_iter = dataset.create_dict_iterator(1, output_numpy=True)
    else:
        ds_iter = prompt_iter
    for step, data in tqdm(enumerate(ds_iter), total=dataset_size):
        file_paths = data["file_path"]
        captions = data["caption"]
        captions = [str(captions[i]) for i in range(len(captions))]
        captions = [text_preprocessing(prompt) for prompt in captions]

        output, output_2 = None, None
        # llm
        if text_encoder is not None:
            output = text_encoder(captions, data_type="video")
            print("D--: ", output.hidden_state)
        # clipL
        if text_encoder_2 is not None:
            output_2 = text_encoder_2(captions, data_type="video")
        save_emb(output, output_2, output_dir, file_paths)

    logger.info(f"Done. Embeddings saved in {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
