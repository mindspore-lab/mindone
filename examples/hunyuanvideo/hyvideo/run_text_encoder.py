import argparse
from typing import List, Union
from pathlib import Path
import csv

import numpy as np
import mindspore as ms

from constants import PROMPT_TEMPLATE, PRECISIONS
from text_encoder import TextEncoder


# prompt_template
prompt_template_name = "dit-llm-encode"
prompt_template_video_name = "dit-llm-encode-video"

prompt_template = PROMPT_TEMPLATE[prompt_template_name] if prompt_template_name is not None else None

# prompt_template_video
prompt_template_video = PROMPT_TEMPLATE[prompt_template_video_name] if prompt_template_video_name is not None else None

# max_length
if prompt_template_video_name is not None:
    crop_start = PROMPT_TEMPLATE[prompt_template_video_name].get("crop_start", 0)
elif prompt_template_name is not None:
    crop_start = PROMPT_TEMPLATE[prompt_template_name].get("crop_start", 0)
else:
    crop_start = 0
text_len = 256
max_length = text_len + crop_start


# args
text_encoder_name = "llm"
tokenizer = "llm"

text_encoder_2_name = "clipL"
tokenizer_2 = "clipL"
text_len_2 = 77

hidden_state_skip_layer = 2
apply_final_norm = False
reproduce = False


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
    parser.add_argument(
        "--text_encoder_precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the text encoder model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="File path of the ckpt of the text encoder.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="File path of the ckpt of the tokenizer.",
    )
    parser.add_argument(
        "--text_encoder_precision_2",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the second text encoder model.",
    )
    parser.add_argument(
        "--text_encoder_2_path",
        type=str,
        default=None,
        help="File path of the ckpt of the second text encoder.",
    )
    parser.add_argument(
        "--tokenizer_2_path",
        type=str,
        default=None,
        help="File path of the ckpt of the second tokenizer.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="File path of prompts, must be a txt or csv file.",
    )
    parser.add_argument("--mode", default=1, type=int, help="Specify the MindSpore mode: 0 for graph mode, 1 for pynative mode")
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
        "--save_dir",
        type=str,
        default="./",
        help="Directory for saving the text encoders' outputs.",
    )

    args = parser.parse_args()

    return args


def build_model(args):
    text_encoder, text_encoder_2 = None, None

    # llm
    if text_encoder_name in args.text_encoder_choices:
        text_encoder = TextEncoder(
            text_encoder_type=text_encoder_name,
            max_length=max_length,
            text_encoder_precision=args.text_encoder_precision,
            text_encoder_path=args.text_encoder_path,
            tokenizer_type=tokenizer,
            tokenizer_path=args.tokenizer_path if args.tokenizer_path is not None else args.text_encoder_path,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=hidden_state_skip_layer,
            apply_final_norm=apply_final_norm,
            reproduce=reproduce,
        )

    # clipL
    if text_encoder_2_name in args.text_encoder_choices:
        text_encoder_2 = TextEncoder(
            text_encoder_type=text_encoder_2_name,
            max_length=text_len_2,
            text_encoder_precision=args.text_encoder_precision_2,
            text_encoder_path=args.text_encoder_2_path,
            tokenizer_type=tokenizer_2,
            tokenizer_path=args.tokenizer_2_path if args.tokenizer_2_path is not None else args.text_encoder_2_path,
            reproduce=reproduce,
        )

    return text_encoder, text_encoder_2


def encode(prompt: Union[List[str], str], args):
    text_encoder, text_encoder_2 = build_model(args)
    output, output_2 = None, None

    # llm
    if text_encoder is not None:
        output = text_encoder(
            prompt,
            use_attention_mask=None,
            output_hidden_states=False,
            do_sample=False,
            hidden_state_skip_layer=None,
            return_texts=False,
            data_type="video",
        )

    # clipL
    if text_encoder_2 is not None:
        output_2 = text_encoder_2(
            prompt,
            use_attention_mask=None,
            output_hidden_states=False,
            do_sample=False,
            hidden_state_skip_layer=None,
            return_texts=False,
            data_type="video",
        )

    return output, output_2


if __name__ == "__main__":
    args = parse_args()
    ms.set_context(mode=args.mode)
    if args.mode == 0:
        ms.set_context(jit_config={"jit_level": args.jit_level})

    if args.prompt_path is None:
        prompt = "hello world"
    else: 
        p = Path(args.prompt_path)
        if not p.exists():
            raise FileNotFoundError(f"The file at path '{args.prompt_path}' does not exist.")
        if p.suffix not in ['.txt', '.csv']:
            raise ValueError(f"The file at path '{args.prompt_path}' is not a.txt or.csv file.")
        if p.suffix == '.txt':
            with p.open('r') as file:
                prompt = file.readlines()
                prompt = [line.rstrip('\n') for line in prompt]
        elif p.suffix == '.csv':
            prompt = []
            with p.open('r') as file:
                reader = csv.reader(file, quotechar='"', quoting=csv.QUOTE_ALL)
                next(reader)    # skip header
                for row in reader:
                    assert len(row) == 2, "The number of columns of csv file should be two."
                    prompt.append(row[1])
        
    output, output_2 = encode(prompt, args)
    if args.save_dir is not None:
        if output is not None:
            np.savez(Path(args.save_dir)/ "llm_output.npz",
                     hidden_state=output.hidden_state.asnumpy(),
                     attention_mask=output.attention_mask.asnumpy())
            print(f"text encoder llm output saved in {args.save_dir}.")
        if output_2 is not None:
            np.savez(Path(args.save_dir)/ "clipL_output.npz",
                     hidden_state=output_2.hidden_state.asnumpy(),
                     attention_mask=output_2.attention_mask.asnumpy())
            print(f"text encoder clipL output saved in {args.save_dir}.")
