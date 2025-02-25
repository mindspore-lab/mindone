import os
import sys
from functools import partial

import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.append(".")
from janus.models.siglip_vit import Block
from janus.utils.io import set_model_param_dtype

np.random.seed(42)


def _diff_res(ms_val, pt_val, eps=1e-8, relax=False):
    if isinstance(ms_val, ms.Tensor):
        ms_val = ms_val.asnumpy()

    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / (np.fabs(pt_val) + eps)

    # relax
    if relax:
        rel_diff = abs_diff / (np.fabs(pt_val))
        tot = np.prod(rel_diff.shape)
        n_nan = np.isnan(rel_diff).sum()
        n_inf = np.isinf(rel_diff).sum()
        print(
            "# values: {}, # nan values: {}, # inf values:{}, (nan+inf)/tot: {}".format(
                tot, n_nan, n_inf, (n_nan + n_inf) / tot
            )
        )
        rel_diff = rel_diff[~np.isnan(rel_diff)]
        rel_diff = rel_diff[~np.isinf(rel_diff)]

    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


def load_from_checkpoint(net, ckpt_path):
    # mainly used in unit test
    parameter_dict = dict()
    if ckpt_path.endswith(".bin"):
        import torch

        sd = torch.load(ckpt_path)
        # filter to keep gen_vision_model params only and remove prefix
        pnames = [p for p in sd]
        for p in pnames:
            # print(p)
            if not ("vision_model.vision_tower.blocks.0." in p):
                sd.pop(p)
            else:
                # remove prefix
                new_pname = p.replace("vision_model.vision_tower.blocks.0.", "")
                sd[new_pname] = sd.pop(p)
        print("Remain params in ckpt: ", len(sd))
        param_dtype = tuple(net.get_parameters())[0].dtype
        for pname in sd:
            # print(pname, sd[pname].shape, sd[pname].dtype)
            np_val = sd[pname].cpu().detach().float().numpy()
            # TODO: support bf16 param loading
            parameter_dict[pname] = ms.Parameter(ms.Tensor(np_val, dtype=param_dtype))

    elif ckpt_path.endswith(".ckpt"):
        parameter_dict = ms.load_checkpoint(ckpt_path)
    else:
        raise ValueError("Unsupported checkpoint format")

    param_not_load, ckpt_not_load = ms.load_param_into_net(net, parameter_dict, strict_load=True)
    print("Net params not load: {}, Total net params not loaded: {}".format(param_not_load, len(param_not_load)))
    print("Ckpt params not load: {}, Total ckpt params not loaded: {}".format(ckpt_not_load, len(ckpt_not_load)))

    return net


def test(pt_np=None, dtype=ms.float32):
    d = 1024
    shape = (1, 576, d)
    if pt_np:
        x = np.load(pt_np[0])
        pt_out = np.load(pt_np[1])
    else:
        x = np.random.normal(size=shape).astype(np.float32)
    x = Tensor(x, dtype)

    # from mindone.diffusers.models.normalization import LayerNorm
    # from mindone.diffusers.models.normalization import FP32LayerNorm as LayerNorm
    from mindspore.mint.nn import LayerNorm

    norm_layer = partial(LayerNorm, eps=1e-6)
    # act_layer = GELU
    # act_layer = partial(nn.GELU, approximate=False)  # 20% mre
    act_layer = partial(nn.GELU, approximate=True)  # mre=0, ok?? why?? doc mismatch???
    # act_layer = partial(mint.nn.GELU, approximate='none')

    net = Block(
        dim=d,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        norm_layer=norm_layer,
        act_layer=act_layer,
    )

    net.set_train(False)
    if dtype != ms.float32:
        set_model_param_dtype(net, dtype=dtype, keep_norm_fp32=False)

    if not os.path.exists("tests/vit_block.ckpt"):
        net = load_from_checkpoint(net, "ckpts/Janus-Pro-1B/pytorch_model.bin")
        ms.save_checkpoint(net, "tests/vit_block.ckpt")
    else:
        net = load_from_checkpoint(net, "tests/vit_block.ckpt")

    out = net(x)

    print(out.shape)
    print(out.sum(), out.std())

    print("ms min max", out.min(), out.max())
    if pt_np:
        print("pt min max: ", pt_out.min(), pt_out.max())
        diff = _diff_res(out.asnumpy(), pt_out, eps=1e-6, relax=True)
        print(diff)


if __name__ == "__main__":
    ms.set_context(mode=1)
    # inp_path = 'tests/pt_vit_block_inp.npy'
    # out_path = 'tests/pt_vit_block_out.npy'
    inp_path = "/home_host/yx/torch_npu/ModelZoo-PyTorch/MindIE/MultiModal/Janus-Pro/pta_vit_block_inp.npy"
    out_path = "/home_host/yx/torch_npu/ModelZoo-PyTorch/MindIE/MultiModal/Janus-Pro/pta_vit_block_out.npy"
    test(pt_np=[inp_path, out_path], dtype=ms.bfloat16)
    # dtype=ms.float32)
