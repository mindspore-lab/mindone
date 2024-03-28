"""
Convert pytorch checkpoint to mindspore checkpoint for inception-i3d.
To run this script, you should have installed both pytorch and mindspore.
pytorch checkpoint can be find in https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt

Usage:

```
python convert_model.py
```

The converted model `inception_i3d.ckpt` will be saved in the same directory as this file belonging to.
"""

import argparse
import os

import torch
from tqdm import tqdm

import mindspore as ms
from mindspore.train.serialization import save_checkpoint


def torch_to_mindspore(pt_ckpt, save=True, save_fp="./inception_i3d.ckpt"):
    pt_param_dict = torch.load(pt_ckpt)
    ms_params = []
    i = 1
    for name in tqdm(pt_param_dict):
        i += 1
        param_dict = {}
        parameter = pt_param_dict[name]
        name = name.replace("Conv3d_1a_7x7", "0", 1)
        name = name.replace("bn.running_mean", "bn.bn2d.moving_mean", 1)
        name = name.replace("bn.running_var", "bn.bn2d.moving_variance", 1)
        name = name.replace("bn.bias", "bn.bn2d.beta", 1)
        name = name.replace("bn.weight", "bn.bn2d.gamma", 1)
        name = name.replace("Conv3d_2b_1x1", "2", 1)
        name = name.replace("Conv3d_2c_3x3", "3", 1)

        name = name.replace("Mixed_3b", "5", 1)
        name = name.replace("Mixed_3c", "6", 1)
        name = name.replace("Mixed_4b", "8", 1)
        name = name.replace("Mixed_4c", "9", 1)
        name = name.replace("Mixed_4d", "10", 1)
        name = name.replace("Mixed_4e", "11", 1)
        name = name.replace("Mixed_4f", "12", 1)
        name = name.replace("Mixed_5b", "14", 1)
        name = name.replace("Mixed_5c", "15", 1)

        param_dict["name"] = name
        param_dict["data"] = ms.Tensor(parameter.numpy())
        ms_params.append(param_dict)

    if save:
        save_checkpoint(ms_params, save_fp)

    return ms_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", type=str, help="pytorch ckpt path")
    args = parser.parse_args()
    # convert to ms checkpoint
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    ckpt_save_fp = os.path.join(__dir__, "inception_i3d.ckpt")
    print("Converting...")
    torch_to_mindspore(args.pt_path, save=True, save_fp=ckpt_save_fp)
    print("Done! Checkpoint saved in ", ckpt_save_fp)


if __name__ == "__main__":
    main()
