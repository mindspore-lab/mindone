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

from mindspore import Tensor
from mindspore import dtype as mstype

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../../"))
sys.path.append(mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from opensora.models.text_encoder import HFEmbedder

from mindone.utils import init_env, set_logger

logger = logging.getLogger(__name__)

Path_dcc = path_type("dcc")  # path to a directory that can be created if it does not exist


def to_numpy(x: Tensor) -> np.ndarray:
    if x.dtype == mstype.bfloat16:
        x = x.astype(mstype.float32)
    return x.asnumpy()


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
                    (output_path / Path(row[column_names[0]]).with_suffix(".npy"), row[column_names[1]])
                    for row in DictReader(file)
                ]
            )
            return paths[rank_id::device_num], captions[rank_id::device_num]
        else:
            captions = [line.strip() for line in file]  # preserve empty lines
            paths = [
                output_path / (f"{i:03d}-" + "-".join(Path(cap).stem.split(" ")[:10]) + ".npy")
                for i, cap in enumerate(captions)
            ]
            return paths[rank_id::device_num], captions[rank_id::device_num]


def main(args):
    save_dir = os.path.abspath(args.output_path)
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    _, rank_id, device_num = init_env(**args.env)

    paths, captions = prepare_captions(args.prompts_file, args.output_path, args.column_names, rank_id, device_num)

    # model initiate and weight loading
    model = HFEmbedder(**args.model)

    # info = (
    #     f"Model name: {args.model_name}\nPrecision: {args.dtype}\nEmbedded sequence length: {args.model_max_length}"
    #     f"\nNumber of devices: {device_num}\nRank ID: {rank_id}\nNumber of captions: {len(captions)}"
    # )
    # logger.info(info)

    for i in trange(0, len(captions), args.batch_size):
        batch = captions[i : i + args.batch_size]
        output = to_numpy(model(batch))

        for j in range(len(output)):
            paths[i + j].parent.mkdir(parents=True, exist_ok=True)
            with open(os.path.join(save_dir, paths[i + j]), "wb") as f:
                np.save(f, output[j])

    logger.info(f"Finished. Embeddings saved to {save_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Text embeddings generation script.")
    parser.add_function_arguments(init_env, "env")
    parser.add_class_arguments(HFEmbedder, "model")
    parser.add_function_arguments(prepare_captions, as_group=False, skip={"rank_id", "device_num"})
    parser.add_argument("--batch_size", default=10, type=int, help="Inference batch size.")
    cfg = parser.parse_args()
    main(cfg)
