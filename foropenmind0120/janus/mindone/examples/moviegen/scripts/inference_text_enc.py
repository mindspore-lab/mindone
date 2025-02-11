import logging
import os
import sys
from csv import DictReader
from pathlib import Path
from typing import List, Tuple

import numpy as np
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr, path_type
from tqdm import trange
from transformers import AutoTokenizer

import mindspore as ms

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.append(mindone_lib_path)
sys.path.append(os.path.join(__dir__, ".."))

from mg.utils import MODEL_DTYPE, to_numpy

from mindone.transformers.models.t5.modeling_t5 import T5EncoderModel
from mindone.utils import init_train_env, set_logger

logger = logging.getLogger(__name__)

Path_dcc = path_type("dcc")  # path to a directory that can be created if it does not exist


def prepare_captions(
    prompts_file: Path_fr,
    output_path: Path_dcc,
    column_names: Tuple[str, str] = ("video", "caption"),
    rank_id: int = 0,
    device_num: int = 1,
) -> Tuple[List[Path], List[str]]:
    """
    Reads prompts from a file and returns a list of saving paths and a list of captions.

    Args:
        prompts_file: Path to the prompt file. Can be a csv file or a txt file.
        output_path: Path to the output directory where the embeddings will be saved.
        column_names: [CSV only] Tuple of column names for video paths and captions.
        rank_id: Current rank id for distributed inference.
        device_num: Number of devices used for distributed inference.

    Returns:
        A tuple containing a list of saving paths and a list of captions.
    """
    prompts_file = prompts_file.absolute
    output_path = Path(output_path.absolute)
    with open(prompts_file, "r", encoding="utf-8") as file:
        if prompts_file.endswith(".csv"):
            paths, captions = zip(
                *[
                    (output_path / Path(row[column_names[0]]).with_suffix(".npz"), row[column_names[1]])
                    for row in DictReader(file)
                ]
            )
            return paths[rank_id::device_num], captions[rank_id::device_num]
        else:
            captions = [line.strip() for line in file]  # preserve empty lines
            paths = [
                output_path / (f"{i:03d}-" + "-".join(Path(cap).stem.split(" ")[:10]) + ".npz")
                for i, cap in enumerate(captions)
            ]
            return paths[rank_id::device_num], captions[rank_id::device_num]


def main(args):
    save_dir = os.path.abspath(args.output_path)
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    _, rank_id, device_num = init_train_env(**args.env)  # TODO: rename as train and infer are identical?

    paths, captions = prepare_captions(args.prompts_file, args.output_path, args.column_names, rank_id, device_num)

    # model initiate and weight loading
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=True, clean_up_tokenization_spaces=False
    )
    model = T5EncoderModel.from_pretrained(
        args.model_name, mindspore_dtype=MODEL_DTYPE[args.dtype.lower()], local_files_only=True
    ).set_train(False)

    info = (
        f"Model name: {args.model_name}\nPrecision: {args.dtype}\nEmbedded sequence length: {args.model_max_length}"
        f"\nNumber of devices: {device_num}\nRank ID: {rank_id}\nNumber of captions: {len(captions)}"
    )
    logger.info(info)

    for i in trange(0, len(captions), args.batch_size):
        batch = captions[i : i + args.batch_size]
        inputs = tokenizer(
            batch,
            max_length=args.model_max_length,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors="np",
        )
        tokens = inputs.input_ids
        masks = inputs.attention_mask
        output = model(ms.Tensor(inputs.input_ids, dtype=ms.int32), ms.Tensor(inputs.attention_mask, dtype=ms.uint8))[0]
        output = to_numpy(output).astype(np.float32)

        for j in range(len(output)):
            paths[i + j].parent.mkdir(parents=True, exist_ok=True)
            with open(os.path.join(save_dir, paths[i + j]), "wb") as f:
                np.savez(f, mask=masks[j], text_emb=output[j], tokens=tokens[j])

    logger.info(f"Finished. Embeddings saved to {save_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Text embeddings generation script.")
    parser.add_function_arguments(init_train_env, "env")
    parser.add_argument("--model_name", type=str, default="google/byt5-small", help="Text encoder model name.")
    parser.add_argument(
        "--dtype", default="fp32", type=str, choices=["fp32", "fp16", "bf16"], help="Text encoder model precision."
    )
    parser.add_function_arguments(prepare_captions, as_group=False, skip={"rank_id", "device_num"})
    parser.add_argument("--batch_size", default=10, type=int, help="Inference batch size.")
    parser.add_argument("--model_max_length", type=int, default=300, help="Model's embedded sequence length.")
    cfg = parser.parse_args()
    main(cfg)
