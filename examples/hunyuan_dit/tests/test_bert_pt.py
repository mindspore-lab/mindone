import numpy as np
import torch
from _common import x


def test_bert_pt(ckpt):
    from transformers import BertModel

    if ckpt is not None:
        net = BertModel.from_pretrained(ckpt, False, revision=None)

    total_params = sum(p.numel() for p in net.parameters())
    print("pt total params: ", total_params)
    print("pt trainable: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    out = net(torch.tensor(x.input_ids), torch.tensor(x.attention_mask))

    print(out[0].size())

    return out[0].detach.numpy()


if __name__ == "__main__":
    out = test_bert_pt("../ckpts/t2i/clip_text_encoder")
    np.save("out_pt_bert.npy", out)
