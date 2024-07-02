import os
import torch
import numpy as np


pytorch_model_paths = [
    "/Users/wyf/.cache/IF_/t5-v1_1-xxl/pytorch_model-00001-of-00002.bin",
    "/Users/wyf/.cache/IF_/t5-v1_1-xxl/pytorch_model-00002-of-00002.bin",
]


for pytorch_model_path in pytorch_model_paths:
    state_dict = torch.load(pytorch_model_path, map_location="cpu")
    state_dict_np = {k: v.numpy() for k, v in state_dict.items()}
    np.save(f"{os.path.splitext(pytorch_model_path)[0]}.npy", state_dict_np)
