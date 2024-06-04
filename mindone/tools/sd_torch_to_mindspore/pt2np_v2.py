import pickle
import sys

import torch

name = sys.argv[1]
ckpt = torch.load(name, map_location="cpu")["state_dict"]

d = {}
with open("mindone/tools/sd_torch_to_mindspore/torch_v2.txt") as file_pt:
    with open("mindone/tools/sd_torch_to_mindspore/mindspore_v2.txt") as file_ms:
        for line_ms, line_pt in zip(file_ms.readlines(), file_pt.readlines()):
            name_pt, _, _ = line_pt.strip().split("#")
            name_ms, _, _ = line_ms.strip().split("#")
            d[name_ms] = ckpt[name_pt].cpu().detach().numpy()
with open("torch.pkl", "wb") as file:
    pickle.dump(d, file)
