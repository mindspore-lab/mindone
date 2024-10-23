import numpy as np
from _common import x

import mindspore as ms


def test_bert_ms(ckpt):
    from mindone.transformers import BertModel

    if ckpt is not None:
        net = BertModel.from_pretrained(ckpt, False, revision=None)

    total_params = sum([param.size for param in net.get_parameters()])
    print("ms total params: ", total_params)
    print("ms trainable: ", sum([param.size for param in net.get_parameters() if param.requires_grad]))

    out = net(ms.tensor(x.input_ids), ms.tensor(x.attention_mask))

    print(out[0].shape)

    return out[0].asnumpy()


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
    ms_out = test_bert_ms("../ckpts/t2i/clip_text_encoder")
    np.save("out_ms_bert.npy", ms_out)

    pt_out = np.load("out_pt_bert.npy")
    print(_diff_res(ms_out, pt_out))
