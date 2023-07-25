import pickle
import sys

import numpy as np
import torch

name = sys.argv[1]
ckpt = torch.load(name, map_location="cpu")["state_dict"]

d = {}
with open("mindone/tools/sd_torch_to_mindspore/torch.txt") as file_pt:
    with open("mindone/tools/sd_torch_to_mindspore/mindspore.txt") as file_ms:
        ms_lines = file_ms.readlines()
        pt_lines = file_pt.readlines()
        i = j = 0
        while i < len(ms_lines):
            line_ms = ms_lines[i]
            name_ms, _, _ = line_ms.strip().split("#")
            if "attn.attn.in_proj" not in line_ms:
                line_pt = pt_lines[j]
                name_pt, _, _ = line_pt.strip().split("#")
                d[name_ms] = ckpt[name_pt].cpu().detach().numpy()
                i += 1
                j += 1
            else:
                w, b = [], []
                for k in range(6):
                    line_pt = pt_lines[j]
                    name_pt, _, _ = line_pt.strip().split("#")
                    j += 1
                    if "weight" in name_pt:
                        w.append(ckpt[name_pt].cpu().detach().numpy())
                    else:
                        b.append(ckpt[name_pt].cpu().detach().numpy())
                d[name_ms] = np.concatenate([b[1], b[0], b[2]])
                i += 1
                line_ms = ms_lines[i]
                name_ms, _, _ = line_ms.strip().split("#")
                d[name_ms] = np.concatenate([w[1], w[0], w[2]])
                i += 1

with open("torch1.pkl", "wb") as file:
    pickle.dump(d, file)
