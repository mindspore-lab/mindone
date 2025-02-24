import argparse
import glob
import logging
import os
from typing import Dict

from fileio import load_safetensors
from tqdm import tqdm

import mindspore as ms
from mindspore import Parameter, Tensor

logger = logging.getLogger(__name__)


def load(root_path: str) -> Dict[str, Tensor]:
    # This method may cause OOM on computer with low memory
    pattern = os.path.join(root_path, "*.safetensors")
    filelist = sorted(glob.glob(pattern))

    filenames = [os.path.basename(x) for x in filelist]
    logger.info(f"Files need to be converted: `{filenames}`")

    params: Dict[str, Tensor] = dict()
    for x in tqdm(filelist, desc="Loading the safetensors"):
        params_chunk = load_safetensors(x)
        if params.keys().isdisjoint(params_chunk.keys()):
            params.update(params_chunk)
        else:
            same_keys = set(params.keys()).intersection(params_chunk.keys())
            raise RuntimeError(f"Duplicated keys found: `{same_keys}`.")
    return params


def ckpt_key_match(checkpoint_data: Dict[str, Parameter]) -> Dict[str, Parameter]:
    ckpt_prefix = "language_model.base_model.model.model"
    model_prefix = "language_model.model"

    aligned_checkpoint = {}
    for key, value in checkpoint_data.items():
        if key.startswith(ckpt_prefix):
            new_key = key.replace(ckpt_prefix, model_prefix, 1)
        else:
            new_key = key

        if "self_attn.q_proj.base_layer.weight" in new_key:
            new_key = new_key.replace("self_attn.q_proj.base_layer.weight", "self_attn.q_proj.weight")
        elif "self_attn.v_proj.base_layer.weight" in new_key:
            new_key = new_key.replace("self_attn.v_proj.base_layer.weight", "self_attn.v_proj.weight")

        # Skip keys with lora parameters for q_proj and v_proj
        if "self_attn.q_proj.lora_" in new_key or "self_attn.v_proj.lora_" in new_key:
            continue

        if "lm_head.weight" in new_key:
            new_key = "language_model.lm_head.weight"

        aligned_checkpoint[new_key] = value

    return aligned_checkpoint


def convert(params: Dict[str, Tensor]) -> Dict[str, Tensor]:
    aligned_params = ckpt_key_match(params)
    return aligned_params


def save(ckpt: Dict[str, Tensor], output: str) -> None:
    output = os.path.abspath(output)
    logger.info(f"Saving to {output}...")
    ms.save_checkpoint(ckpt, output)
    logger.info(f"Saving to {output}...Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert PLLaVa checkpoints into Mindspore Format")
    parser.add_argument("src", help="Directory storing the safetensors")
    parser.add_argument("-o", "--output", help="Name of the output Mindspore checkpoint")

    args = parser.parse_args()

    params = load(args.src)
    params = convert(params)
    save(params, args.output)


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s: %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)
    main()
