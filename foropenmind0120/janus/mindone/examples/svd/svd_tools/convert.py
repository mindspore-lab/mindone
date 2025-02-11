import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr, path_type
from omegaconf import OmegaConf
from safetensors.torch import load_file

from mindspore import Parameter, load_param_into_net, save_checkpoint

sys.path.append(os.path.join(os.path.dirname(__file__), "../../stable_diffusion_xl"))  # FIXME: remove in the future
from gm.helpers import create_model

Path_dcc = path_type("dcc")  # path to a directory that can be created if it does not exist


def convert(pt_weights_file: Path_fr, config: Path_fr, out_dir: Optional[Path_dcc] = None):
    """
    Convert PyTorch weights to MindSpore format.

    Args:
        pt_weights_file: Path to the PyTorch weights file.
        config: Path to the VideoLDM config file.
        out_dir: (optional) Path to directory where the converted checkpoint will be saved.
                If not provided, the converted checkpoint will be saved in the same directory as the PyTorch checkpoint.
    Raises:
        ValueError: If some parameters were not loaded during the conversion process.
    """
    pt_weights_file = Path(pt_weights_file)
    pt_ckpt = load_file(pt_weights_file)
    pt_keys = list(pt_ckpt.keys())

    config = OmegaConf.load(config.absolute)
    network, _ = create_model(config, freeze=True)
    ms_weights = network.parameters_dict()
    ms_keys = list(ms_weights.keys())

    # after sorting, PyTorch and MindSpore parameters are aligned
    pt_keys = sorted(pt_keys)
    ms_keys = sorted(ms_keys)

    for pt_key, ms_key in zip(pt_keys, ms_keys):
        ms_weights[ms_key] = Parameter(np.array(pt_ckpt[pt_key]), name=ms_key)

    param_not_load, _ = load_param_into_net(network, ms_weights)
    if param_not_load:
        raise ValueError(f"Something went wrong. The following parameters were not loaded:\n{param_not_load}")

    out_dir = pt_weights_file.parent / "ms_models" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir / (pt_weights_file.stem + ".ckpt")

    save_checkpoint(network, str(out_dir))
    print(f"Conversion completed. Checkpoint is saved to:\n{out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_function_arguments(convert)
    cfg = parser.parse_args()
    convert(**cfg.as_dict())
