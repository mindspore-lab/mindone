import sys

import numpy as np
from _common import inputs, kwargs

import mindspore as ms


def test_hunyuan_dit_ms(ckpt):
    ms_code_path = "PATH/mindone/examples/hunyuan_dit/"
    sys.path.insert(0, ms_code_path)
    from hydit.config import get_args
    from hydit.modules.models import HUNYUAN_DIT_CONFIG, HunYuanDiT
    from hydit.utils.tools import convert_state_dict, load_state_dict

    if ckpt is not None:
        model_config = HUNYUAN_DIT_CONFIG["DiT-XL/2"]
        args = get_args()
        model = HunYuanDiT(args, **model_config).half()

    state_dict = load_state_dict(ckpt)
    state_dict = convert_state_dict(model, state_dict)
    local_state = {k: v for k, v in model.parameters_and_names()}
    for k, v in state_dict.items():
        if k in local_state:
            v.set_dtype(local_state[k].dtype)
        else:
            pass  # unexpect key keeps origin dtype
    ms.load_param_into_net(model, state_dict, strict_load=True)

    total_params = sum([param.size for param in model.get_parameters()])
    print("ms total params: ", total_params)
    print("ms trainable: ", sum([param.size for param in model.get_parameters() if param.requires_grad]))

    inputs = [ms.tensor(input) for input in inputs]
    kwargs = {k: ms.Tensor(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
    out = model(*inputs, **kwargs)[0]

    print(out.shape)

    return out.asnumpy()


def _diff_res(ms_val, pt_val, eps=1e-8):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / np.fabs(pt_val).mean()
    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


if __name__ == "__main__":
    ms.set_context(mode=0)
    ms_out = test_hunyuan_dit_ms("../ckpts/t2i/model/pytorch_model_distill.safetensors")
    np.save("out_ms_hunyuan_dit.npy", ms_out)

    pt_out = np.load("out_pt_hunyuan_dit.npy")
    print(_diff_res(ms_out, pt_out))
