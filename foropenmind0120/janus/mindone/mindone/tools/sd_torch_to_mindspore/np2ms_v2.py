import pickle
import sys

import mindspore as ms

name = sys.argv[1]

with open("torch.pkl", "rb") as file:
    d = pickle.load(file)

ckpt = []
with open("mindone/tools/sd_torch_to_mindspore/torch_v2.txt") as file_pt:
    with open("mindone/tools/sd_torch_to_mindspore/mindspore_v2.txt") as file_ms:
        for line_ms, line_pt in zip(file_ms.readlines(), file_pt.readlines()):
            name_ms, _, _ = line_ms.strip().split("#")
            data = d[name_ms]
            ckpt.append({"name": name_ms, "data": ms.Tensor(data)})

ms.save_checkpoint(ckpt, f"ms_{name}.ckpt")
