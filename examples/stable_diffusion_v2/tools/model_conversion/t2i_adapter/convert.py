import sys
from pathlib import Path

import torch

from mindspore import Parameter, load_param_into_net, save_checkpoint

sys.path.append("examples/stable_diffusion_v2")

from examples.stable_diffusion_v2.adapters import get_adapter


def convert(pt_weights_file: Path):
    task = pt_weights_file.stem.split("_")[1]  # PyTorch's checkpoints have t2iadapter_TASK_sdXXXX.pth names

    pt_ckpt = torch.load(pt_weights_file, map_location="cpu")
    pt_keys = list(pt_ckpt.keys())

    if task == "style":  # swap last two layers `ln_post` and `ln_pre`
        pt_keys[38:] = pt_keys[40:42] + pt_keys[38:40]
    elif task != "color":  # move the input convolution from the end to the beginning for the full adapter architecture
        pt_keys = pt_keys[-2:] + pt_keys[:-2]

    network = get_adapter(task)
    ms_weights = network.parameters_dict()
    ms_keys = list(ms_weights.keys())

    for pt_key, ms_key in zip(pt_keys, ms_keys):
        ms_weights[ms_key] = Parameter(pt_ckpt[pt_key].numpy(), name=ms_key)

    param_not_load, _ = load_param_into_net(network, ms_weights)
    if param_not_load:
        raise ValueError(f"Something went wrong. The following parameters were not loaded:\n{param_not_load}")

    output_ckpt = pt_weights_file.absolute().parents[1] / "ms_models" / (pt_weights_file.stem + ".ckpt")
    save_checkpoint(network, str(output_ckpt))
    print(f"Conversion completed. Checkpoint is saved to:\n{output_ckpt}")


if __name__ == "__main__":
    path = Path(sys.argv[1])
    convert(path)
