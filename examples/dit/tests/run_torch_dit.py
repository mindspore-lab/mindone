import os
import sys

import numpy as np
import torch

TORCH_PATH = "./DiT"  # the directory to https://github.com/facebookresearch/DiT
sys.path.append(os.path.abspath(TORCH_PATH))
from models import DiT_models


def load_pt_dit(model_name="DiT-XL/2", dtype="fp16", dit_checkpoint="models/DiT-XL-2-256x256.pt", device="cuda"):
    image_size = int(dit_checkpoint.split(".")[0].split("-")[-1].split("x")[-1])
    latent_size = image_size // 8
    dit_model = DiT_models[model_name](
        input_size=latent_size,
        num_classes=1000,
    ).to(device)

    if dit_checkpoint:
        state_dict = torch.load(dit_checkpoint, weights_only=True, map_location="cpu")
        dit_model.load_state_dict(state_dict)
    else:
        print("Initialize DIT randomly")
    dit_model.eval()
    return dit_model


def init_inputs(image_size, device="cuda"):
    latent_size = image_size // 8
    bs = 2
    num_channels = 4
    x = torch.randn(bs, num_channels, latent_size, latent_size)
    y = torch.randint(0, 2, (bs,))
    t = torch.arange(bs)
    # save the inputs to .npz
    np.savez("pt_inputs.npz", x=x.numpy(), y=y.numpy(), t=t.numpy())
    # send to device
    x, y, t = x.to(device), y.to(device), t.to(device)
    return x, y, t


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y, t = init_inputs(256, device)
    dit_model = load_pt_dit(device=device)
    output = dit_model(x, y, t)
    print(output.shape)
    np.save("pt_output.npy", output.cpu().detach().numpy())
