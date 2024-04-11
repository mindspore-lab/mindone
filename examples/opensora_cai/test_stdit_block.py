import os, sys
import torch
import mindspore as ms
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.models.stdit import STDiTBlock

ms.set_context(mode=1)

hidden_size = 1024
T = num_temporal = 1
S = num_spatial = 16*16  # num_patches // self.num_temporal

args = dict(
        hidden_size=hidden_size,
        num_heads=8,
        d_s=S,
        d_t=T,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
    )

B, N, C = 2, T*S, hidden_size 
x = np.random.normal(size=(B, N, C)).astype(np.float32)

# condition, text, 
N_tokens = 64
y = np.random.normal(size=(1, B*N_tokens, C)).astype(np.float32)
y_lens = [N_tokens] * B

tpe = None

# time embedding
t = np.random.normal(size=(B, 6*C)).astype(np.float32)

global_inputs = (x, y, t)


def test_net_ms(x, ckpt=None, net_class=None, args=None):
    net_ms = net_class(**args)
    net_ms.set_train(False)
    if ckpt:
       sd = ms.load_checkpoint(ckpt)
       m, u = ms.load_param_into_net(net_ms, sd)
       print("net param not loaded: ", m)
       print("ckpt param not loaded: ", u)
    if isinstance(x, (list, tuple)):
        inputs = []
        for xi in x:
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

def test_net_pt(x, ckpt=None, save_ckpt_fn=None, net_class=None, args=None):
    net_pt = net_class(**args).cuda()
    net_pt.eval()
    if ckpt is not None:
        checkpoint = torch.load(ckpt)
        net_pt.load_state_dict(checkpoint['model_state_dict'])

    if save_ckpt_fn:
        torch.save({'model_state_dict': net_pt.state_dict(),
                    }, f"tests/{save_ckpt_fn}.pth")

    if isinstance(x, (list, tuple)):
        inputs = []
        for xi in x:
            inputs.append(torch.Tensor(xi).cuda())
        res_pt = net_pt(*inputs)
    else:
        res_pt = net_pt(torch.Tensor(x))

    total_params = sum(p.numel() for p in net_pt.parameters())
    print("pt total params: ", total_params)
    print("pt trainable: ", sum(p.numel() for p in net_pt.parameters() if p.requires_grad))
    print(res_pt.shape)
    return res_pt.detach().cpu().numpy(), net_pt

def _convert_ckpt(pt_ckpt):
    # sd = torch.load(pt_ckpt, map_location="CPU")['model_state_dict']
    sd = torch.load(pt_ckpt)['model_state_dict']
    target_data = []

    # import pdb
    # pdb.set_trace()

    for k in sd:
        if '.' not in k:
            # only for GroupNorm
            ms_name = k.replace("weight", "gamma").replace("bias", "beta")
        else:
            if 'norm' in k:
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
    sys.path.append(pt_code_path)
    from opensora.models.stdit.stdit import STDiTBlock as STD_PT

    ckpt_fn = 'stdb'
    pt_res, net_pt = test_net_pt(global_inputs, save_ckpt_fn=ckpt_fn, net_class=STD_PT, args=args)
    print("pt out range: ", pt_res.min(), pt_res.max())

    ckpt = _convert_ckpt(f"tests/{ckpt_fn}.pth")

    ms_res, net_ms = test_net_ms(global_inputs, ckpt=ckpt, net_class=STDiTBlock, args=args)
    print(_diff_res(ms_res, pt_res))
    # (0.0001554184, 0.0014244393)


def test_stdit_raw():

    # model = STDiT(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)

    net = STDiTBlock(**args)
    net.set_train(False)

    B, N, C = 1, T*S, hidden_size 
    x = np.random.normal(size=(B, N, C)).astype(np.float32)

    # condition, text, 
    N_tokens = 64
    y = np.random.normal(size=(1, B*N_tokens, C)).astype(np.float32)
    y_lens = [N_tokens] * B

    tpe = None

    # time embedding
    t = np.random.normal(size=(B, 6*C)).astype(np.float32)

    out = net(ms.Tensor(x), ms.Tensor(y), ms.Tensor(t), mask=None, tpe=None)
    print(out.shape)


if __name__ == "__main__":
    ms.set_context(mode=1)
    compare_stdit()


