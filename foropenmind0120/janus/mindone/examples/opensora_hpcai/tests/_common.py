import os

import numpy as np

use_mask = True
print("use mask: ", use_mask)

# data config
hidden_size = 1152

text_emb_dim = 4096
max_tokens = 120

num_frames = 16
image_size = 256

vae_t_compress = 1
vae_s_compress = 8
vae_out_channels = 4

text_emb_dim = 4096
max_tokens = 120

input_size = (num_frames // vae_t_compress, image_size // vae_s_compress, image_size // vae_s_compress)
B, C, T, H, W = 2, vae_out_channels, input_size[0], input_size[1], input_size[2]

npz = "input_256.npz"

if npz is not None and os.path.exists(npz):
    print("=> Load inputs from", npz)
    d = np.load(npz)
    x, y = d["x"], d["y"]
    mask = d["mask"]
    mask = np.repeat(mask, x.shape[0] // mask.shape[0], axis=0)

    # t = np.random.randint(low=0, high=1000, size=B).astype(np.float32)
    t = np.ones(B).astype(np.float32) * 999

else:
    print(
        "WARNING: You should use the real data as inputs to test the accuracy! Please save input_256.npz by pdb instead!"
    )
    x = np.random.normal(size=(B, C, T, H, W)).astype(np.float32)
    t = np.random.randint(low=0, high=1000, size=B).astype(np.float32)
    # condition, text,
    y = np.random.normal(size=(B, 1, max_tokens, text_emb_dim)).astype(np.float32)
    y_lens = np.random.randint(low=110, high=max_tokens, size=[B])

    # mask (B, max_tokens)
    mask = np.zeros(shape=[B, max_tokens]).astype(np.int8)  # TODO: use bool?
    for i in range(B):
        mask[i, : y_lens[i]] = np.ones(y_lens[i]).astype(np.int8)

    print("input x, y: ", x.shape, y.shape)
    print("mask: ", mask.shape)

    np.savez(npz, x=x, y=y, mask=mask)

if not use_mask:
    mask = None

# model config
model_extra_args = dict(
    input_size=input_size,
    in_channels=vae_out_channels,
    caption_channels=text_emb_dim,
    model_max_length=max_tokens,
    space_scale=0.5,
    time_scale=1.0,
)
