import sys
import os
import numpy as np

sys.path.append(".")
import numpy as np
import torch
from vae.vae_temporal import CausalConv3d, ResBlock, Encoder, Decoder

import mindspore as ms

bs, cin, T, H, W = 1, 9, 8, 256, 256
cout = 16  # hidden size, actually 128
x = np.random.normal(size=(bs, cin, T, H, W))


def test_cconv3d():
    ms.set_context(mode=0)
    conv_in = CausalConv3d(cin, cout, kernel_size=3, stride=1, padding=1)
    xo = conv_in(ms.Tensor(x, dtype=ms.float32))

    print("out shape: ", xo.shape)


def _diff_res(ms_val, pt_val, eps=1e-8):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / (np.fabs(pt_val) + eps)
    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


def compare_cconv3d(copy_weights=True):
    import torch
    # pt_code_path = "/Users/Samit/Data/Work/HW/ms_kit/aigc/Open-Sora"
    # sys.path.append(pt_code_path)
    # from opensora.models.vae.vae_temporal import CausalConv3d as CConv3d_PT

    from pt_vae_temporal import CausalConv3d as CConv3d_PT

    cc3_pt = CConv3d_PT(cin, cout, kernel_size=3)
    out_pt = cc3_pt(torch.Tensor(x))
    out_pt = out_pt.detach().numpy()
    print("PT: ", out_pt.shape, out_pt.mean(), out_pt.sum())

    if copy_weights:
        torch.save(
            {
                "model_state_dict": cc3_pt.state_dict(),
            },
            "cc3.pth",
        )

        target_data = []
        for k in cc3_pt.state_dict():
            target_data.append({"name": k, "data": ms.Tensor(cc3_pt.state_dict()[k].detach().numpy())})
        ms.save_checkpoint(target_data, "cc3.ckpt")

    cc3_ms = CausalConv3d(cin, cout, kernel_size=3)

    if copy_weights:
        ms.load_checkpoint("cc3.ckpt", net=cc3_ms)

    out_ms = cc3_ms(ms.Tensor(x, dtype=ms.float32))
    out_ms = out_ms.asnumpy()
    print("MS: ", out_ms.shape, out_ms.mean(), out_ms.sum())
    print("Diff: ", _diff_res(out_pt, out_ms))


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


def compare_ResBlock(pdb_debug=False, backend='ms'):
    # pt_code_path = "/data3/hyx/Open-Sora-Plan-cc41/"
    # sys.path.append(pt_code_path)
    # from opensora.models.ae.videobase.causal_vae.modeling_causalvae import Encoder as Enc_PT
    from pt_vae_temporal import ResBlock as Res_PT

    # NOTE: after vae 2d, h w compressed to 1/8, z=4
    bs, cin, T, H, W = 1, 128, 9, 64, 64
    # cout = 16
    # net_kwargs = dict(in_channels=cin, out_channels=cout)

    inp = 'res_input.npy'
    if os.path.exists(inp):
        print('load input from ', inp)
        x = np.load(inp)
    else:
        x = np.random.normal(size=(bs, cin, T, H, W))
        np.save(inp, x)
    print("Input sum: ", x.sum())

    kwargs = dict(in_channels=128, filters=128)
    net_pt = Res_PT(**kwargs)
    # import torch
    # net_pt = torch.nn.Conv3d(**net_kwargs,kernel_size= (3, 3, 3), stride=(2,1,1), padding=(0,1,1))
    # net_pt = torch.nn.Conv3d(**net_kwargs,kernel_size= (3, 3, 3), stride=(1,1,1), padding=0)

    name = 'res'
    pt_ckpt = f'{name}.pth'
    if os.path.exists(pt_ckpt):
        print('load ckpt from ', pt_ckpt)
        net_pt.load_state_dict(torch.load(pt_ckpt))
    else:
        torch.save(net_pt.state_dict(), pt_ckpt)

    if pdb_debug and backend=='pt':
        import pdb; pdb.set_trace()
    res_pt = net_pt(torch.Tensor(x))
    res_pt = res_pt.detach().numpy()
    print("PT output shape: ", res_pt.shape)

    ms_ckpt = _convert_ckpt(pt_ckpt, name=name)
    ms.set_context(mode=0)
    net_ms = ResBlock(**kwargs)
    # net_ms = ms.nn.Conv3d(**net_kwargs, kernel_size=(3,3,3), stride=(2,1,1), pad_mode="pad", padding=(0,0,1,1,1,1), has_bias=True)
    # net_ms = ms.nn.Conv3d(**net_kwargs, kernel_size=(3,3,3), stride=(1,1,1), pad_mode="valid", has_bias=True)
    ms.load_checkpoint(ms_ckpt, net_ms)

    if pdb_debug and backend=='ms':
        import pdb; pdb.set_trace()
    res_ms = net_ms(ms.Tensor(x, dtype=ms.float32))
    res_ms = res_ms.asnumpy()

    print("MS output shape: ", res_ms.shape)
    print("Diff: ", _diff_res(res_ms, res_pt))


def compare_Encoder(pdb_debug=False, backend='ms'):
    # pt_code_path = "/data3/hyx/Open-Sora-Plan-cc41/"
    # sys.path.append(pt_code_path)
    # from opensora.models.ae.videobase.causal_vae.modeling_causalvae import Encoder as Enc_PT
    from pt_vae_temporal import Encoder as Enc_PT

    # NOTE: after vae 2d, h w compressed to 1/8, z=4
    bs, cin, T, H, W = 1, 4, 9, 64, 64
    latent_embed_dim=4
    # net_kwargs = dict(in_channels=cin, out_channels=cout)

    inp = 'enc_input.npy'
    if os.path.exists(inp):
        print('load input from ', inp)
        x = np.load(inp)
    else:
        x = np.random.normal(size=(bs, cin, T, H, W))
        np.save(inp, x)
    print("Input sum: ", x.sum())

    net_pt = Enc_PT(latent_embed_dim=latent_embed_dim*2)

    name = 'enc'
    pt_ckpt = f'{name}.pth'
    if os.path.exists(pt_ckpt):
        print('load ckpt from ', pt_ckpt)
        net_pt.load_state_dict(torch.load(pt_ckpt))
    else:
        torch.save(net_pt.state_dict(), pt_ckpt)

    if pdb_debug and backend=='pt':
        import pdb; pdb.set_trace()
    res_pt = net_pt(torch.Tensor(x))
    res_pt = res_pt.detach().numpy()
    print("PT output shape: ", res_pt.shape)

    ms_ckpt = _convert_ckpt(pt_ckpt, name=name)
    ms.set_context(mode=0)
    net_ms = Encoder(latent_embed_dim=latent_embed_dim*2)
    ms.load_checkpoint(ms_ckpt, net_ms)

    if pdb_debug and backend=='ms':
        import pdb; pdb.set_trace()
    res_ms = net_ms(ms.Tensor(x, dtype=ms.float32))
    res_ms = res_ms.asnumpy()

    print("MS output shape: ", res_ms.shape)
    print("Diff: ", _diff_res(res_ms, res_pt))


def compare_Decoder(pdb_debug=False, backend='ms'):
    # pt_code_path = "/data3/hyx/Open-Sora-Plan-cc41/"
    # sys.path.append(pt_code_path)
    # from opensora.models.ae.videobase.causal_vae.modeling_causalvae import Encoder as Enc_PT
    from pt_vae_temporal import Decoder as Dec_PT

    latent_embed_dim = 4
    bs, cin, T, H, W = 1, latent_embed_dim, 2, 64, 64
    # net_kwargs = dict(in_channels=cin, out_channels=cout)

    inp = 'dec_input.npy'
    if os.path.exists(inp):
        print('load input from ', inp)
        x = np.load(inp)
    else:
        x = np.random.normal(size=(bs, cin, T, H, W))
        np.save(inp, x)
    print("Input sum: ", x.sum())

    net_pt = Dec_PT(latent_embed_dim=latent_embed_dim)

    name = 'dec'
    pt_ckpt = f'{name}.pth'
    if os.path.exists(pt_ckpt):
        print('load ckpt from ', pt_ckpt)
        net_pt.load_state_dict(torch.load(pt_ckpt))
    else:
        torch.save(net_pt.state_dict(), pt_ckpt)

    if pdb_debug and backend=='pt':
        import pdb; pdb.set_trace()
    res_pt = net_pt(torch.Tensor(x))
    res_pt = res_pt.detach().numpy()
    print("PT output shape: ", res_pt.shape)

    ms_ckpt = _convert_ckpt(pt_ckpt, name=name)
    ms.set_context(mode=0)
    net_ms = Decoder(latent_embed_dim=latent_embed_dim)
    ms.load_checkpoint(ms_ckpt, net_ms)

    if pdb_debug and backend=='ms':
        import pdb; pdb.set_trace()
    res_ms = net_ms(ms.Tensor(x, dtype=ms.float32))
    res_ms = res_ms.asnumpy()

    print("MS output shape: ", res_ms.shape)
    print("Diff: ", _diff_res(res_ms, res_pt))


if __name__ == "__main__":
    # compare_cconv3d()
    # compare_ResBlock()
    # compare_Encoder()
    compare_Decoder()

