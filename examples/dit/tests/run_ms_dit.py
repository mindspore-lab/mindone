import os
import sys

import numpy as np

import mindspore as ms
from mindspore import mint

sys.path.insert(0, ".")
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

from utils.model_utils import load_dit_ckpt_params

from mindone.models.dit import DiT_models
from mindone.utils.amp import auto_mixed_precision


def load_ms_dit(model_name="DiT-XL/2", dtype="fp16", dit_checkpoint="models/DiT-XL-2-256x256.ckpt"):
    image_size = int(dit_checkpoint.split(".")[0].split("-")[-1].split("x")[-1])
    latent_size = image_size // 8
    dit_model = DiT_models[model_name](
        input_size=latent_size,
        num_classes=1000,
        block_kwargs={"enable_flash_attention": True},
    )

    if dtype == "fp16":
        model_dtype = ms.float16
        dit_model = auto_mixed_precision(dit_model, amp_level="O2", dtype=model_dtype)
    elif dtype == "bf16":
        model_dtype = ms.bfloat16
        dit_model = auto_mixed_precision(dit_model, amp_level="O2", dtype=model_dtype)
    else:
        model_dtype = ms.float32

    if dit_checkpoint:
        dit_model = load_dit_ckpt_params(dit_model, dit_checkpoint)
    else:
        print("Initialize DIT ramdonly")
    dit_model = dit_model.set_train(False)
    for param in dit_model.get_parameters():  # freeze dit_model
        param.requires_grad = False
    return dit_model


def init_inputs(image_size):
    latent_size = image_size // 8
    bs = 2
    num_channels = 4
    x = mint.randn(bs, num_channels, latent_size, latent_size)
    y = mint.randint(0, 2, (bs,))
    t = mint.arange(bs)
    # save the inputs to .npz
    np.savez("ms_inputs.npz", x=x.asnumpy(), y=y.asnumpy(), t=t.asnumpy())
    return x, y, t


def load_inputs(pt_inputs="./pt_inputs.npz"):
    pt_inputs = np.load(pt_inputs)
    x = mint.Tensor(pt_inputs["x"])
    y = mint.Tensor(pt_inputs["y"])
    t = mint.Tensor(pt_inputs["t"])
    return x, y, t


if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE)
    # x,y,t = init_inputs(256)
    x, y, t = load_inputs(pt_inputs="./pt_inputs.npz")
    dit_model = load_ms_dit()
    output = dit_model(x, y, t)
    print(output.shape)
    np.save("ms_output.npy", output.asnumpy())
