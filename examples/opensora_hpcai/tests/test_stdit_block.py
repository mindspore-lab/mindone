import os
import sys

import numpy as np
import torch

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

sys.path.append("./")
from opensora.models.stdit.stdit import STDiTBlock

from mindone.utils.amp import auto_mixed_precision

use_mask = False

# input args
hidden_size = 1152
num_head = 16  # head_dim=72

max_tokens = 120

T = num_temporal = 16
S = num_spatial = 16 * 16  # 256x256

args = dict(
    hidden_size=hidden_size,
    num_heads=num_head,
    d_s=S,
    d_t=T,
    mlp_ratio=4.0,
    drop_path=0.0,
    enable_flashattn=False,
    enable_layernorm_kernel=False,
    enable_sequence_parallelism=False,
)

B, N, C = 2, T * S, hidden_size

fp = "tests/stdit_block_inp.npz"


def get_inputs(npz=None):
    if npz is not None and os.path.exists(npz):
        data = np.load(npz)
        x, y, t = data["x"], data["y"], data["t"]
        if use_mask:
            mask = data["mask"]
    else:
        x = np.random.normal(size=(B, N, C)).astype(np.float32)

        y = np.random.normal(size=(1, B * max_tokens, C)).astype(np.float32)
        y_lens = np.random.randint(low=max_tokens // 2, high=max_tokens, size=[B])
        print("y_lens: ", y_lens)

        # time embedding
        t = np.random.normal(size=(B, 6 * C)).astype(np.float32)

        save_dict = dict(x=x, y=y, t=t)

        # mask (B, max_tokens)
        if use_mask:
            mask = np.zeros(shape=[B, max_tokens]).astype(np.uint8)
            for i in range(B):
                mask[i, : y_lens[i]] = np.ones(y_lens[i])
        else:
            mask = None

        if mask is not None:
            save_dict["mask"] = mask

        np.savez(fp, **save_dict)
        print("inputs saved in", fp)

    if not use_mask:
        mask = None

    tpe = None
    return x, y, t, mask, tpe


x, y, t, mask, tpe = get_inputs(fp)

global_inputs = (x, y, t, mask, tpe)


def test_net_ms(x, ckpt=None, net_class=None, args=None):
    net_ms = net_class(**args)
    net_ms.set_train(False)

    net_ms = auto_mixed_precision(net_ms, "O2", ms.float16)

    if ckpt:
        sd = ms.load_checkpoint(ckpt)
        m, u = ms.load_param_into_net(net_ms, sd)
        print("net param not loaded: ", m)
        print("ckpt param not loaded: ", u)
    if isinstance(x, (list, tuple)):
        inputs = []
        for xi in x:
            if xi is not None:
                inputs.append(ms.Tensor(xi, dtype=ms.float32))
        res_ms = net_ms(*inputs)
    else:
        res_ms = net_ms(ms.Tensor(x, dtype=ms.float32))
    total_params = sum([param.size for param in net_ms.get_parameters()])
    total_trainable = sum([param.size for param in net_ms.get_parameters() if param.requires_grad])
    print("ms total params: ", total_params)
    print("ms trainable: ", total_trainable)

    print(res_ms.shape)
    return res_ms.asnumpy(), net_ms


def torch_mask_convert(mask, y):
    # mask: [B, n_tokens], y: (1, B*N_token, C)

    _B = mask.shape[0]
    y = y.reshape((_B, -1, y.shape[-1]))
    y = y.reshape((_B, 1, -1, y.shape[-1]))

    # y: text, [B, 1, N_token, C]
    if mask.shape[0] != y.shape[0]:
        mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
    mask = mask.squeeze(1).squeeze(1)
    y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
    y_lens = mask.sum(dim=1).tolist()

    return y_lens


# inputs only for stdit block
def test_net_pt(x, y, t, mask=None, tpe=None, ckpt=None, save_ckpt_fn=None, net_class=None, args=None):
    #
    net_pt = net_class(**args).cuda()
    net_pt.eval()
    if ckpt is not None and os.path.exists(ckpt):
        checkpoint = torch.load(ckpt)
        net_pt.load_state_dict(checkpoint["model_state_dict"])
    else:
        if save_ckpt_fn:
            torch.save(
                {
                    "model_state_dict": net_pt.state_dict(),
                },
                f"tests/{save_ckpt_fn}.pth",
            )

    x = torch.Tensor(x).cuda()
    y = torch.Tensor(y).cuda()
    t = torch.Tensor(t).cuda()
    if mask is not None:
        mask = torch.Tensor(mask).cuda()
        mask = torch_mask_convert(mask, y)
        print("converted mask for pt: ", mask)
    if tpe is not None:
        tpe = torch.Tensor(tpe).cuda()

    res_pt = net_pt(x, y, t, mask, tpe)

    total_params = sum(p.numel() for p in net_pt.parameters())
    print("pt total params: ", total_params)
    print("pt trainable: ", sum(p.numel() for p in net_pt.parameters() if p.requires_grad))
    print(res_pt.shape)
    return res_pt.detach().cpu().numpy(), net_pt


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
        target_data.append({"name": ms_name, "data": ms.Tensor(sd[k].detach().cpu().numpy())})

    save_fn = pt_ckpt.replace(".pth", ".ckpt")
    ms.save_checkpoint(target_data, save_fn)

    return save_fn


def _diff_res(ms_val, pt_val):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()
    return mae, max_ae


def compare_stdit():
    pt_code_path = "/home/mindocr/yx/Open-Sora/"
    # pt_code_path = "/srv/hyx/Open-Sora/"
    sys.path.append(pt_code_path)
    from opensora.models.stdit.stdit.stdit import STDiTBlock as STD_PT

    ckpt_fn = "stdb"
    ckpt_fp_pt = f"tests/{ckpt_fn}.pth"
    pt_res, net_pt = test_net_pt(x, y, t, mask, tpe, ckpt=ckpt_fp_pt, save_ckpt_fn=ckpt_fn, net_class=STD_PT, args=args)
    print("pt out range: ", pt_res.min(), pt_res.max())

    ckpt = _convert_ckpt(f"tests/{ckpt_fn}.pth")

    ms_res, net_ms = test_net_ms(global_inputs, ckpt=ckpt, net_class=STDiTBlock, args=args)
    print(_diff_res(ms_res, pt_res))
    # (0.0001554184, 0.0014244393)


def test_stdit_ms():
    # model = STDiT(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)

    args["enable_flashattn"] = False
    # args['use_recompute'] = False

    net = STDiTBlock(**args)
    net.set_train(False)

    net = auto_mixed_precision(net, "O2", ms.float16)

    B, N, C = 1, T * S, hidden_size
    x = np.random.normal(size=(B, N, C)).astype(np.float32)

    # condition, text,
    max_tokens = 120
    y = np.random.normal(size=(1, B * max_tokens, C)).astype(np.float32)
    # y_lens = [max_tokens] * B

    # tpe = None

    # time embedding
    t = np.random.normal(size=(B, 6 * C)).astype(np.float32)

    out = net(ms.Tensor(x), ms.Tensor(y), ms.Tensor(t), mask=None, tpe=None)
    print(out.shape)


if __name__ == "__main__":
    ms.set_context(mode=1)
    compare_stdit()
    # test_stdit_ms()
