# flake8: noqa
import os
import sys

import numpy as np

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.utils.amp import auto_mixed_precision

sys.path.append("../")
from _common import *
from opensora.models.layers.blocks import Attention, LayerNorm
from opensora.models.stdit.stdit import STDiT_XL_2  # STDiTBlock


def test_stdit(ckpt_path=None, amp=None):
    model_extra_args["enable_flashattn"] = False
    model_extra_args["use_recompute"] = False
    model_extra_args["patchify_conv3d_replace"] = "conv2d"

    net = STDiT_XL_2(**model_extra_args)
    net.set_train(False)

    if ckpt_path is not None:
        net.load_from_checkpoint(ckpt_path)
    if amp is not None:
        print("use AMP", amp)
        if amp == "O2":
            net = auto_mixed_precision(
                net, "O2", ms.float16, custom_fp32_cells=[LayerNorm, Attention]
            )  # , nn.GELU, nn.SiLU])
        else:
            net = auto_mixed_precision(net, "O1")

    total_params = sum([param.size for param in net.get_parameters()])
    total_trainable = sum([param.size for param in net.get_parameters() if param.requires_grad])
    print("ms total params: ", total_params)
    print("ms trainable: ", total_trainable)

    if use_mask:
        out = net(ms.Tensor(x), ms.Tensor(t), ms.Tensor(y), mask=ms.Tensor(mask))
    else:
        out = net(ms.Tensor(x), ms.Tensor(t), ms.Tensor(y))

    print(out.shape)
    print(out.sum(), out.std())

    return out.asnumpy()


def _diff_res(ms_val, pt_val, eps=1e-8):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / (np.fabs(pt_val) + eps)
    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


if __name__ == "__main__":
    ms.set_context(mode=0)
    ms_out = test_stdit("../models/OpenSora-v1-HQ-16x256x256.ckpt")
    np.save("out_ms_stdit.npy", ms_out)

    pt_out = np.load("out_pt_stdit.npy")
    print(_diff_res(ms_out, pt_out))
