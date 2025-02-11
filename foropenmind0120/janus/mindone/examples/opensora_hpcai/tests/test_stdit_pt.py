# flake8: noqa
"""
Usage:
cd tests
python test_stdit_pt.py
"""
import os
import sys

import numpy as np
import torch
from _common import *

"""
x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
timestep (torch.Tensor): diffusion time steps; of shape [B]
y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
"""


def test_stdit_pt(ckpt):
    pt_code_path = "/srv/hyx/Open-Sora/"
    sys.path.insert(0, pt_code_path)
    from opensora.models.stdit.stdit import STDiT_XL_2 as STD_PT

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


if __name__ == "__main__":
    out = test_stdit_pt("../models/OpenSora-v1-HQ-16x256x256.pth")
    np.save("out_pt_stdit.npy", out)
