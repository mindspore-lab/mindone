# TODO: use trained checkpoint and real data to check.

import sys

import torch

sys.path.append(".")
import numpy as np
from ae.models.causal_vae_3d import CausalVAEModel, Decoder, Encoder
from ae.models.modules import SpatialDownsample2x, SpatialUpsample2x, TimeDownsample2x, TimeUpsample2x

import mindspore as ms

z_channels = 4

args = dict(
    ch=128,
    out_ch=3,
    ch_mult=(1, 2, 4, 4),
    num_res_blocks=1,
    attn_resolutions=[16],
    dropout=0.0,
    resamp_with_conv=True,
    in_channels=3,
    resolution=256,
    z_channels=z_channels,
    double_z=True,
    use_linear_attn=False,
    attn_type="vanilla3D",  # diff 3d
    time_compress=2,  # diff 3d
    split_time_upsample=True,
)

ae_args = dict(
    ddconfig=args,
    embed_dim=z_channels,
)

bs, cin, T, H, W = 1, 3, 9, 256, 256
x = np.random.normal(size=(bs, cin, T, H, W))


def test_net_ms(x, ckpt=None, net_class=Encoder, args=None):
    net_ms = net_class(**args)
    net_ms.set_train(False)
    if ckpt:
        sd = ms.load_checkpoint(ckpt)
        m, u = ms.load_param_into_net(net_ms, sd)
        print("net param not loaded: ", m)
        print("ckpt param not loaded: ", u)

    res_ms = net_ms(ms.Tensor(x, dtype=ms.float32))
    total_params = sum([param.size for param in net_ms.get_parameters()])
    print("ms total params: ", total_params)

    print(res_ms.shape)
    return res_ms.asnumpy(), net_ms


def test_net_pt(x, ckpt=None, save_ckpt_fn=None, net_class=None, args=None):
    net_pt = net_class(**args)
    net_pt.eval()
    if ckpt is not None:
        checkpoint = torch.load(ckpt)
        net_pt.load_state_dict(checkpoint["model_state_dict"])

    if save_ckpt_fn:
        torch.save(
            {
                "model_state_dict": net_pt.state_dict(),
            },
            f"tests/{save_ckpt_fn}.pth",
        )

    res_pt = net_pt(torch.Tensor(x))

    total_params = sum(p.numel() for p in net_pt.parameters())
    print("pt total params: ", total_params)
    print(res_pt.shape)

    return res_pt.detach().numpy(), net_pt


def _convert_ckpt(pt_ckpt):
    # sd = torch.load(pt_ckpt, map_location="CPU")['model_state_dict']
    sd = torch.load(pt_ckpt)["model_state_dict"]
    target_data = []

    # import pdb
    # pdb.set_trace()

    for k in sd:
        if "." not in k:
            # only for GroupNorm
            ms_name = k.replace("weight", "gamma").replace("bias", "beta")
        else:
            if "norm" in k:
                ms_name = k.replace(".weight", ".gamma").replace(".bias", ".beta")
            else:
                ms_name = k
        target_data.append({"name": ms_name, "data": ms.Tensor(sd[k].detach().numpy())})

    save_fn = pt_ckpt.replace(".pth", ".ckpt")
    ms.save_checkpoint(target_data, save_fn)

    return save_fn


def _diff_res(ms_val, pt_val):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()
    return mae, max_ae


def test_encoder():
    ms_res, net_ms = test_net_ms(x, ckpt=None, net_class=Encoder, args=args)
    print(ms_res.shape)


def test_decoder():
    z_shape = (1, z_channels, 3, 32, 32)  # b c t h w
    # z_shape = (1, z_channels, 2, 32, 32)  # b c t h w
    z = np.random.normal(size=z_shape)
    ms_res, net_ms = test_net_ms(z, ckpt=None, net_class=Decoder, args=args)
    print(ms_res.shape)


def compare_time_down2x(ckpt_fn="timedown2x"):
    # pt_code_path = "/data3/hyx/Open-Sora-Plan-cc41/"
    pt_code_path = "/home/mindocr/yx/Open-Sora-Plan/"
    sys.path.append(pt_code_path)
    from opensora.models.ae.videobase.modules.updownsample import TimeDownsample2x as TD_PT

    args = dict(kernel_size=3)
    pt_res, net_pt = test_net_pt(x, save_ckpt_fn=None, net_class=TD_PT, args=args)
    # ckpt = _convert_ckpt(f"tests/{ckpt_fn}.pth")

    ms_res, net_ms = test_net_ms(x, ckpt=None, net_class=TimeDownsample2x, args=args)
    print(_diff_res(ms_res, pt_res))


def compare_space_down(ckpt_fn="spacedown"):
    d = 64
    x = np.random.normal(size=(bs, d, T, H, W))

    args = dict(chan_in=d, chan_out=d)
    ms_res, net_ms = test_net_ms(x, ckpt=None, net_class=SpatialDownsample2x, args=args)


def compare_space_up(ckpt_fn="spaceup"):
    d = 64
    x = np.random.normal(size=(bs, d, T, H, W))

    args = dict(chan_in=d, chan_out=d)
    ms_res, net_ms = test_net_ms(x, ckpt=None, net_class=SpatialUpsample2x, args=args)


def compare_time_up(ckpt_fn="timeup"):
    d = 64
    x = np.random.normal(size=(bs, d, T, H, W))

    args = dict(exclude_first_frame=True)
    ms_res, net_ms = test_net_ms(x, ckpt=None, net_class=TimeUpsample2x, args=args)


def compare_encoder(x, ckpt_fn="encoder", backend="ms+pt"):
    if isinstance(x, str):
        x = np.load(x)
    else:
        save_fn = "tests/encoder_inp.npy"
        np.save(save_fn, x)
        print("saved random data in ", save_fn)

    if "pt" in backend:
        # pt_code_path = "/data3/hyx/Open-Sora-Plan-cc41/"
        pt_code_path = "/home/mindocr/yx/Open-Sora-Plan/"
        sys.path.append(pt_code_path)
        from opensora.models.ae.videobase.causal_vae.modeling_causalvae import Encoder as Encoder_PT

        # import pdb
        # pdb.set_trace()

        pt_res, net_pt = test_net_pt(x, save_ckpt_fn=ckpt_fn, net_class=Encoder_PT, args=args)
        print("pt out range: ", pt_res.min(), pt_res.max())

        ckpt = _convert_ckpt(f"tests/{ckpt_fn}.pth")
    else:
        ckpt = f"tests/{ckpt_fn}.ckpt"

    if "ms" in backend:
        # import pdb
        # pdb.set_trace()

        ms_res, net_ms = test_net_ms(x, ckpt=ckpt, net_class=Encoder, args=args)

    if "pt" in backend and "ms" in backend:
        print(_diff_res(ms_res, pt_res))
        # (0.0001554184, 0.0014244393)


def compare_decoder():
    z_shape = (1, z_channels, 2, 32, 32)  # b c t h w
    z = np.random.normal(size=z_shape)

    pt_code_path = "/data3/hyx/Open-Sora-Plan-cc41/"
    sys.path.append(pt_code_path)
    from opensora.models.ae.videobase.causal_vae.modeling_causalvae import Decoder as Decoder_PT

    ckpt_fn = "decoder"
    pt_res, net_pt = test_net_pt(z, save_ckpt_fn=ckpt_fn, net_class=Decoder_PT, args=args)
    print("pt out range: ", pt_res.min(), pt_res.max())

    ckpt = _convert_ckpt(f"tests/{ckpt_fn}.pth")

    ms_res, net_ms = test_net_ms(z, ckpt=ckpt, net_class=Decoder, args=args)
    print(_diff_res(ms_res, pt_res))
    # (0.0001554184, 0.0014244393)


def test_vae3d():
    bs, cin, T, H, W = 1, 3, 9, 256, 256
    x = np.random.normal(size=(bs, cin, T, H, W))

    # ms_res, net_ms = test_net_ms(x, ckpt=None, net_class=CausalVAEModel, args=ae_args)
    ae = CausalVAEModel(ddconfig=args, embed_dim=4)
    ae.set_train(False)
    # res_ms = net(ms.Tensor(x, dtype=ms.float32))

    z = ae.encode(ms.Tensor(x, dtype=ms.float32))
    print("z shape: ", z.shape)

    recon = ae.decode(z)
    print("recon shape: ", recon.shape)


if __name__ == "__main__":
    ms.set_context(mode=0)

    # test_encoder()
    # test_decoder()
    compare_encoder(x)
    # compare_decoder()
    # test_vae3d()

    # compare_encoder("tests/encoder_inp.npy", backend='pt')
    # compare_encoder("tests/encoder_inp.npy", backend='ms')

    # compare_time_down2x()
    # compare_space_down()
    # compare_space_up()
    # compare_time_up()
