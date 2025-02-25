import os
import sys

import numpy as np

import mindspore as ms
from mindspore import Tensor, amp

sys.path.append(".")
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../..")))  # for mindone

# default setup, unit load hard to load ckpt this way, do entire model loading
from janus.models.siglip_vit import create_siglip_vit
from utils import diff_res

np.random.seed(42)


def test(dtype=ms.bfloat16):
    jp1b = "/mnt/disk2/fredhong/hf_ckpts/Janus-Pro-1B/pytorch_model.bin"

    # load input and golden gt, run this testing under Janus dir
    input_tensor = Tensor(np.load("./image_tensor.npy")).to(dtype)
    gt_tensor = np.load("./image_forward_outs.npy")
    # print(input_tensor)
    print(f"gt tensor dtype is {gt_tensor.dtype}")

    model_name: str = "siglip_large_patch16_384"
    select_layer: int = -1
    vision_tower = create_siglip_vit(
        model_name,
        select_layer=select_layer,
    )
    vision_tower.load_from_checkpoint(jp1b)

    print(f"dtype conversion is using with {dtype}")

    # if dtype != ms.float32:
    #     set_model_param_dtype(vision_tower, dtype=dtype, keep_norm_fp32=False)
    if dtype != ms.float32:
        amp.auto_mixed_precision(vision_tower, amp_level="O2", dtype=dtype)

    # cal & eval
    out = vision_tower(input_tensor)
    out = out.to(ms.float32).asnumpy()

    # assert np.allclose(out, gt_tensor), f"recal result is not closed to gt!, out:{out.shape}\n{out}\ngt:{gt_tensor.shape}\n{gt_tensor}"

    diff = diff_res(out, gt_tensor)
    print(diff)
    print("test finish")


if __name__ == "__main__":
    ms.set_context(device_id=7, mode=1, pynative_synchronize=True)
    test()
