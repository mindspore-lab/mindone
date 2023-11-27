from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from adapters import get_adapter
from jsonargparse import ArgumentParser
from safetensors.torch import load_file

from mindspore import Parameter, load_param_into_net, save_checkpoint


def convert(diffusion_model: Literal["sd", "sdxl"], pt_weights_file: Path, task: str, out_dir: Optional[Path] = None):
    """
    Convert PyTorch weights to MindSpore format.

    Args:
        diffusion_model: Stable Diffusion model version. Either "sd" or "sdxl".
        pt_weights_file: Path to the PyTorch weights file.
        task: The task for which the weights are being converted.
        out_dir: (optional) Path to directory where the converted checkpoint will be saved.
                If not provided, the converted checkpoint will be saved in the same directory as the PyTorch checkpoint.
    Raises:
        ValueError: If some parameters were not loaded during the conversion process.
    """
    if pt_weights_file.suffix == ".safetensors":
        pt_ckpt = load_file(pt_weights_file)
        pt_keys = list(pt_ckpt.keys())
        pt_keys[::2], pt_keys[1::2] = pt_keys[1::2], pt_keys[::2]  # weights and biases are swapped in safetensors
    else:
        pt_ckpt = torch.load(pt_weights_file, map_location="cpu")
        pt_keys = list(pt_ckpt.keys())

    if task == "style":  # swap last two layers `ln_post` and `ln_pre`
        pt_keys[38:] = pt_keys[40:42] + pt_keys[38:40]
    elif task != "color":  # move the input convolution from the end to the beginning for the full adapter architecture
        pt_keys = pt_keys[-2:] + pt_keys[:-2]

    network = get_adapter(diffusion_model, task)
    ms_weights = network.parameters_dict()
    ms_keys = list(ms_weights.keys())

    for pt_key, ms_key in zip(pt_keys, ms_keys):
        ms_weights[ms_key] = Parameter(np.array(pt_ckpt[pt_key]), name=ms_key)

    param_not_load, _ = load_param_into_net(network, ms_weights)
    if param_not_load:
        raise ValueError(f"Something went wrong. The following parameters were not loaded:\n{param_not_load}")

    if out_dir is None:
        out_dir = pt_weights_file.parent / "ms_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir / (pt_weights_file.stem + ".ckpt")

    save_checkpoint(network, str(out_dir))
    print(f"Conversion completed. Checkpoint is saved to:\n{out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_function_arguments(convert)
    cfg = parser.parse_args()
    convert(**cfg.as_dict())
