import os
import sys

import numpy as np
import torch

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

sys.path.append("./")
from opensora.models.stdit.stdit import STDiT_XL_2  # , STDiTBlock

from mindone.utils.amp import auto_mixed_precision

"""
x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
timestep (torch.Tensor): diffusion time steps; of shape [B]
y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
"""

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

npz = "input_256.npz"

if npz is not None:
    d = np.load(npz)
    x, y = d["x"], d["y"]
    mask = d["mask"]
    mask = np.repeat(mask, x.shape[0] // mask.shape[0], axis=0)

    # TODO: fix it
    t = np.random.randint(low=0, high=1000, size=B).astype(np.float32)

else:
    x = np.random.normal(size=(B, C, T, H, W)).astype(np.float32)
    t = np.random.randint(low=0, high=1000, size=B).astype(np.float32)
    # condition, text,
    y = np.random.normal(size=(B, 1, max_tokens, text_emb_dim)).astype(np.float32)
    y_lens = np.random.randint(low=4, high=max_tokens, size=[B])

    # mask (B, max_tokens)
    mask = np.zeros(shape=[B, max_tokens]).astype(np.int8)  # TODO: use bool?
    for i in range(B):
        mask[i, : y_lens[i]] = np.ones(y_lens[i]).astype(np.int8)

    print("input x, y: ", x.shape, y.shape)
    print("mask: ", mask.shape)

if not use_mask:
    mask = None

# model config
model_extra_args = dict(
    input_size=input_size,
    in_channels=vae_out_channels,
    caption_channels=text_emb_dim,
    model_max_length=max_tokens,
)


def test_stdit(ckpt_path=None, amp=False):
    model_extra_args["enable_flashattn"] = False
    model_extra_args["use_recompute"] = False

    net = STDiT_XL_2(**model_extra_args)
    net.set_train(False)

    if ckpt_path is not None:
        sd = ms.load_checkpoint(ckpt_path)
        m, u = ms.load_param_into_net(net, sd)
        print("net param not load: ", m)
        print("ckpt param not load: ", u)

    if amp:
        print("use AMP")
        net = auto_mixed_precision(net, "O2", ms.float16)

    total_params = sum([param.size for param in net.get_parameters()])
    total_trainable = sum([param.size for param in net.get_parameters() if param.requires_grad])
    print("ms total params: ", total_params)
    print("ms trainable: ", total_trainable)

    # for param in net.get_parameters():
    #    # if param.requires_grad:
    #    print(param.name, tuple(param.shape))

    if use_mask:
        out = net(ms.Tensor(x), ms.Tensor(t), ms.Tensor(y), mask=ms.Tensor(mask))
    else:
        out = net(ms.Tensor(x), ms.Tensor(t), ms.Tensor(y))

    print(out.shape)
    print(out.mean(), out.std())

    return out.asnumpy()


def test_stdit_pt(ckpt):
    pt_code_path = "/srv/hyx/Open-Sora/"
    sys.path.append(pt_code_path)
    from opensora.models.stdit.stdit.stdit import STDiT_XL_2 as STD_PT

    net = STD_PT(**model_extra_args).cuda()
    net.eval()

    if ckpt is not None:
        sd = torch.load(ckpt)
        net.load_state_dict(sd)

    total_params = sum(p.numel() for p in net.parameters())
    print("pt total params: ", total_params)
    print("pt trainable: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    # for pname, p in net.named_parameters():
    #    # if p.requires_grad:
    #    print(pname, tuple(p.shape))

    if use_mask:
        out = net(
            torch.Tensor(x).cuda(), torch.Tensor(t).cuda(), torch.Tensor(y).cuda(), mask=torch.Tensor(mask).cuda()
        )
    else:
        out = net(torch.Tensor(x).cuda(), torch.Tensor(t).cuda(), torch.Tensor(y).cuda())

    print(out.shape)

    return out.detach().cpu().numpy()


def _diff_res(ms_val, pt_val):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()
    return mae, max_ae


def compare_stdit():
    pt_ckpt = "models/OpenSora-v1-HQ-16x256x256.pth"
    pt_out = test_stdit_pt(pt_ckpt)

    ms_ckpt = "models/OpenSora-v1-HQ-16x256x256.ckpt"
    ms_out = test_stdit(ms_ckpt)

    print(_diff_res(ms_out, pt_out))
    # (5.4196875e-07, 1.5079975e-05)


if __name__ == "__main__":
    ms.set_context(mode=0)
    # test_stdit_pt('models/OpenSora-v1-HQ-16x256x256.pth')
    # test_stdit('models/OpenSora-v1-HQ-16x256x256.ckpt')
    compare_stdit()
