"""
Convert pytorch checkpoint to mindspore checkpoint for inception v3.
To run this script, you should have installed both pytorch and mindspore.

Usage:

```
python convert_model.py
```

The converted model `inception_v3_fid.ckpt` will be saved in the same directory as this file belonging to.
"""

import os

import torch
from tqdm import tqdm

import mindspore as ms
from mindspore.train.serialization import save_checkpoint

from examples.stable_diffusion_v2.eval.fid.utils import download_model

PT_FID_WEIGHTS_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"  # noqa: E501


def torch_to_mindspore(pt_ckpt, save=True, save_fp="./inception_v3_fid.ckpt"):
    pt_param_dict = torch.load(pt_ckpt)
    ms_params = []
    i = 1
    for name in tqdm(pt_param_dict):
        if name.find("num_batches_tracked") != -1:
            continue
        # print(i, name)
        i += 1
        param_dict = {}
        parameter = pt_param_dict[name]
        name = name.replace("_3x3.", ".", 1)
        name = name.replace("running_mean", "moving_mean", 1)
        name = name.replace("running_var", "moving_variance", 1)
        name = name.replace("bn.bias", "bn.beta", 1)
        name = name.replace("bn.weight", "bn.gamma", 1)
        name = name.replace("_1x1.", ".", 1)
        name = name.replace("branch1x1", "branch0", 1)
        name = name.replace("branch5x5_1", "branch1.0", 1)
        name = name.replace("branch5x5_2", "branch1.1", 1)
        name = name.replace("branch_pool", "branch_pool.1", 1)
        name = name.replace("branch3x3.", "branch0.", 1)
        name = name.replace("branch7x7_1", "branch1.0", 1)
        name = name.replace("branch7x7_2", "branch1.1", 1)
        name = name.replace("branch7x7_3", "branch1.2", 1)
        name = name.replace("branch7x7dbl_1", "branch2.0", 1)
        name = name.replace("branch7x7dbl_2", "branch2.1", 1)
        name = name.replace("branch7x7dbl_3", "branch2.2", 1)
        name = name.replace("branch7x7dbl_4", "branch2.3", 1)
        name = name.replace("branch7x7dbl_5", "branch2.4", 1)
        if name.find("Mixed_7b") != -1 or name.find("Mixed_7c") != -1:
            name = name.replace("branch3x3_1", "branch1", 1)
        else:
            name = name.replace("branch3x3_1", "branch0.0", 1)
        if name.find("Mixed_6a") != -1:
            name = name.replace("branch3x3dbl_1", "branch1.0", 1)
            name = name.replace("branch3x3dbl_2", "branch1.1", 1)
            name = name.replace("branch3x3dbl_3.", "branch1.2.", 1)
        else:
            name = name.replace("branch3x3dbl_1", "branch2.0", 1)
            name = name.replace("branch3x3dbl_2", "branch2.1", 1)
            name = name.replace("branch3x3dbl_3.", "branch2.2.", 1)
        name = name.replace("branch3x3_2.", "branch0.1.", 1)
        name = name.replace("branch3x3_2a", "branch1_a", 1)
        name = name.replace("branch3x3_2b", "branch1_b", 1)
        name = name.replace("branch3x3dbl_3a", "branch2_a", 1)
        name = name.replace("branch3x3dbl_3b", "branch2_b", 1)

        name = name.replace("branch7x7x3_1", "branch1.0", 1)
        name = name.replace("branch7x7x3_2", "branch1.1", 1)
        name = name.replace("branch7x7x3_3", "branch1.2", 1)
        name = name.replace("branch7x7x3_4", "branch1.3", 1)
        param_dict["name"] = name
        param_dict["data"] = ms.Tensor(parameter.numpy())
        ms_params.append(param_dict)

    if save:
        save_checkpoint(ms_params, save_fp)

    return ms_params


def main():
    # download torch checkpoint
    pt_fp = download_model(PT_FID_WEIGHTS_URL)

    # convert to ms checkpoint
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    ckpt_save_fp = os.path.join(__dir__, "inception_v3_fid.ckpt")
    print("Converting...")
    torch_to_mindspore(pt_fp, save=True, save_fp=ckpt_save_fp)
    print("Done! Checkpoint saved in ", ckpt_save_fp)


if __name__ == "__main__":
    main()
