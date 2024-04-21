import os
import sys

import numpy as np

import mindspore as ms
from mindspore import nn

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.utils.amp import auto_mixed_precision

sys.path.append("./")
from opensora.models.layers.blocks import Attention, LayerNorm
from opensora.models.stdit import STDiT_XL_2  # STDiTBlock

use_mask = True
print("use mask: ", use_mask)

# data config
hidden_size = 1152

text_emb_dim = 4096
max_tokens = 120

num_frames = 16
image_size = 256

vae_t_compress = 1
vae_s_compress = 8
vae_out_channels = 4

text_emb_dim = 4096
max_tokens = 120

input_size = (num_frames // vae_t_compress, image_size // vae_s_compress, image_size // vae_s_compress)
B, C, T, H, W = 2, vae_out_channels, input_size[0], input_size[1], input_size[2]

npz = "tests/stdit_inp_rand.npz"

if npz is not None and os.path.exists(npz):
    d = np.load(npz)
    x, y = d["x"], d["y"]
    mask = d["mask"]
    mask = np.repeat(mask, x.shape[0] // mask.shape[0], axis=0)

    t = np.ones(B).astype(np.float32) * 999
else:
    x = np.random.normal(size=(B, C, T, H, W)).astype(np.float32)
    # t = np.random.randint(low=0, high=1000, size=B).astype(np.float32)
    t = np.ones(B).astype(np.float32) * 999

    # condition, text,
    y = np.random.normal(size=(B, 1, max_tokens, text_emb_dim)).astype(np.float32)
    y_lens = np.random.randint(low=4, high=max_tokens, size=[B])

    # mask (B, max_tokens)
    mask = np.zeros(shape=[B, max_tokens]).astype(np.int8)  # TODO: use bool?
    for i in range(B):
        mask[i, : y_lens[i]] = np.ones(y_lens[i]).astype(np.int8)

    print("input x, y: ", x.shape, y.shape)
    print("mask: ", mask.shape)

    np.savez(npz, x=x, y=y, mask=mask)
    print("inputs saved in", npz)

if not use_mask:
    mask = None

# model config
model_extra_args = dict(
    input_size=input_size,
    in_channels=vae_out_channels,
    caption_channels=text_emb_dim,
    model_max_length=max_tokens,
)


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
            net = auto_mixed_precision(net, "O2", ms.float16, custom_fp32_cells=[LayerNorm, Attention]) # , nn.GELU, nn.SiLU])
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

def _diff_res(ms_val, pt_val):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()
    return mae, max_ae



if __name__ == "__main__":
    ms.set_context(mode=0)
    out_fp32 = test_stdit("models/OpenSora-v1-HQ-16x256x256.ckpt")
    out_o2 = test_stdit("models/OpenSora-v1-HQ-16x256x256.ckpt", amp="O2")
    
    print(_diff_res(out_o2, out_fp32))
