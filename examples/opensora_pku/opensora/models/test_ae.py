import os
import sys

sys.path.append(".")
import numpy as np
import torch

# from ae.models.modules import AttnBlock3D, CausalConv3d, ResnetBlock3D

mindone_dir = "/home/mindocr/yx/mindone"
sys.path.insert(0, mindone_dir)
from ae.videobase.causal_vae.modeling_causalvae_v2 import Decoder, Encoder
from ae.videobase.modules.updownsample import TimeDownsampleRes2x

import mindspore as ms

bs, cin, T, H, W = 1, 3, 8, 256, 256
cout = 16  # hidden size, actually 128
x = np.random.normal(size=(bs, cin, T, H, W))


def _diff_res(ms_val, pt_val, eps=1e-8):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / (np.fabs(pt_val) + eps)
    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


def _convert_ckpt(pt_ckpt, name, contain_gn=False):
    target_data = []
    sd = torch.load(pt_ckpt)
    for k in sd:
        if "." not in k and contain_gn:
            # only for GroupNorm
            ms_name = k.replace("weight", "gamma").replace("bias", "beta")
        else:
            if "norm" in k:
                ms_name = k.replace(".weight", ".gamma").replace(".bias", ".beta")
            else:
                ms_name = k
        target_data.append({"name": ms_name, "data": ms.Tensor(sd[k].detach().numpy())})

    save_fn = f"{name}.ckpt"
    ms.save_checkpoint(target_data, save_fn)
    return save_fn


def compare_TimeDownsampleRes2x(pdb_debug=False, backend="ms"):
    # pt_code_path = "/data3/hyx/Open-Sora-Plan-cc41/"
    pt_code_path = "/home/mindocr/yx/Open-Sora-Plan/"

    sys.path.append(pt_code_path)
    from opensora.models.ae.videobase.modules.updownsample import TimeDownsampleRes2x as TDR_PT

    bs, cin, T, H, W = 1, 16, 8, 256, 256
    cout = 16
    net_kwargs = dict(in_channels=cin, out_channels=cout)

    inp = "tdr_input.npy"
    if os.path.exists(inp):
        print("load input from ", inp)
        x = np.load(inp)
    else:
        x = np.random.normal(size=(bs, cin, T, H, W))
        np.save(inp, x)
    print("Input sum: ", x.sum())

    net_pt = TDR_PT(**net_kwargs)
    # import torch
    # net_pt = torch.nn.Conv3d(**net_kwargs,kernel_size= (3, 3, 3), stride=(2,1,1), padding=(0,1,1))
    # net_pt = torch.nn.Conv3d(**net_kwargs,kernel_size= (3, 3, 3), stride=(1,1,1), padding=0)

    name = "tdr"
    pt_ckpt = f"{name}.pth"
    if os.path.exists(pt_ckpt):
        print("load ckpt from ", pt_ckpt)
        net_pt.load_state_dict(torch.load(pt_ckpt))
    else:
        torch.save(net_pt.state_dict(), pt_ckpt)

    # if pdb_debug and backend == "pt":
    #     import pdb

    #     pdb.set_trace()
    res_pt = net_pt(torch.Tensor(x))
    res_pt = res_pt.detach().numpy()

    ms_ckpt = _convert_ckpt(pt_ckpt, name=name)
    # ms.set_context(mode=0)
    net_ms = TimeDownsampleRes2x(**net_kwargs, replace_avgpool3d=True)
    # net_ms = ms.nn.Conv3d(**net_kwargs, kernel_size=(3,3,3), stride=(2,1,1), pad_mode="pad", padding=(0,0,1,1,1,1), has_bias=True)
    # net_ms = ms.nn.Conv3d(**net_kwargs, kernel_size=(3,3,3), stride=(1,1,1), pad_mode="valid", has_bias=True)
    ms.load_checkpoint(ms_ckpt, net_ms)

    # if pdb_debug and backend == "ms":
    #     import pdb

    #     pdb.set_trace()
    res_ms = net_ms(ms.Tensor(x, dtype=ms.float32))
    res_ms = res_ms.asnumpy()

    print("Diff: ", _diff_res(res_ms, res_pt))


def compare_Encoder(pdb_debug=False, backend="ms"):
    # pt_code_path = "/data3/hyx/Open-Sora-Plan-cc41/"
    pt_code_path = "/home/mindocr/yx/Open-Sora-Plan/"

    sys.path.append(pt_code_path)
    from opensora.models.ae.videobase.causal_vae.modeling_causalvae import Encoder as Enc_PT

    bs, cin, T, H, W = 1, 3, 9, 256, 256
    # cout = 16
    # net_kwargs = dict(in_channels=cin, out_channels=cout)

    inp = "enc_input.npy"
    if os.path.exists(inp):
        print("load input from ", inp)
        x = np.load(inp)
    else:
        x = np.random.normal(size=(bs, cin, T, H, W))
        np.save(inp, x)
    print("Input sum: ", x.sum())

    net_pt = Enc_PT()
    # import torch
    # net_pt = torch.nn.Conv3d(**net_kwargs,kernel_size= (3, 3, 3), stride=(2,1,1), padding=(0,1,1))
    # net_pt = torch.nn.Conv3d(**net_kwargs,kernel_size= (3, 3, 3), stride=(1,1,1), padding=0)

    name = "enc"
    pt_ckpt = f"{name}.pth"
    if os.path.exists(pt_ckpt):
        print("load ckpt from ", pt_ckpt)
        net_pt.load_state_dict(torch.load(pt_ckpt))
    else:
        torch.save(net_pt.state_dict(), pt_ckpt)

    # if pdb_debug and backend == "pt":
    #     import pdb

    #     pdb.set_trace()
    res_pt = net_pt(torch.Tensor(x))
    res_pt = res_pt.detach().numpy()

    ms_ckpt = _convert_ckpt(pt_ckpt, name=name)
    ms.set_context(mode=0)
    net_ms = Encoder()
    # net_ms = ms.nn.Conv3d(**net_kwargs, kernel_size=(3,3,3), stride=(2,1,1), pad_mode="pad", padding=(0,0,1,1,1,1), has_bias=True)
    # net_ms = ms.nn.Conv3d(**net_kwargs, kernel_size=(3,3,3), stride=(1,1,1), pad_mode="valid", has_bias=True)
    ms.load_checkpoint(ms_ckpt, net_ms)

    # if pdb_debug and backend == "ms":
    #     import pdb

    #     pdb.set_trace()
    res_ms = net_ms(ms.Tensor(x, dtype=ms.float32))
    res_ms = res_ms.asnumpy()

    print("Diff: ", _diff_res(res_ms, res_pt))


def compare_Decoder(pdb_debug=False, backend="ms"):
    # pt_code_path = "/data3/hyx/Open-Sora-Plan-cc41/"
    pt_code_path = "/home/mindocr/yx/Open-Sora-Plan/"

    sys.path.append(pt_code_path)
    from opensora.models.ae.videobase.causal_vae.modeling_causalvae import Decoder as Dec_PT

    z_channels = 4
    z_shape = (1, z_channels, 3, 32, 32)  # b c t h w
    # cout = 16
    # net_kwargs = dict(in_channels=cin, out_channels=cout)

    inp = "dec_input.npy"
    if os.path.exists(inp):
        print("load input from ", inp)
        x = np.load(inp)
    else:
        x = np.random.normal(size=z_shape)
        np.save(inp, x)
    print("Input sum: ", x.sum())

    net_pt = Dec_PT()
    # import torch
    # net_pt = torch.nn.Conv3d(**net_kwargs,kernel_size= (3, 3, 3), stride=(2,1,1), padding=(0,1,1))
    # net_pt = torch.nn.Conv3d(**net_kwargs,kernel_size= (3, 3, 3), stride=(1,1,1), padding=0)

    name = "dec"
    pt_ckpt = f"{name}.pth"
    if os.path.exists(pt_ckpt):
        print("load ckpt from ", pt_ckpt)
        net_pt.load_state_dict(torch.load(pt_ckpt))
    else:
        torch.save(net_pt.state_dict(), pt_ckpt)

    # if pdb_debug and backend == "pt":
    #     import pdb

    #     pdb.set_trace()
    res_pt = net_pt(torch.Tensor(x))
    res_pt = res_pt.detach().numpy()

    ms_ckpt = _convert_ckpt(pt_ckpt, name=name)
    ms.set_context(mode=0)
    net_ms = Decoder()
    # net_ms = ms.nn.Conv3d(**net_kwargs, kernel_size=(3,3,3), stride=(2,1,1), pad_mode="pad", padding=(0,0,1,1,1,1), has_bias=True)
    # net_ms = ms.nn.Conv3d(**net_kwargs, kernel_size=(3,3,3), stride=(1,1,1), pad_mode="valid", has_bias=True)
    ms.load_checkpoint(ms_ckpt, net_ms)

    # if pdb_debug and backend == "ms":
    #     import pdb

    #     pdb.set_trace()
    res_ms = net_ms(ms.Tensor(x, dtype=ms.float32))
    res_ms = res_ms.asnumpy()

    print("Diff: ", _diff_res(res_ms, res_pt))


if __name__ == "__main__":
    # compare_TimeDownsampleRes2x(pdb_debug=False, backend='pt')
    # compare_Encoder(pdb_debug=False, backend='pt')
    compare_Decoder(pdb_debug=False, backend="pt")
