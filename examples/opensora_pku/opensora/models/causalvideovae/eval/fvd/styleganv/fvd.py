import math
import os

from mindspore import context, export, load, mint, nn, ops

try:
    import torch
except ImportError:
    print(
        "For the first-time running, torch is required to load torchscript model and convert to onnx, but import torch leads to an ImportError!"
    )

# https://github.com/universome/fvd-comparison


def load_i3d_pretrained(bs=1):
    i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i3d_torchscript.pt")
    onnx_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i3d_torchscript.onnx")
    mindir_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i3d_torchscript.mindir")
    if not os.path.exists(mindir_filepath):
        if not os.path.exists(filepath):
            print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
            os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
        if not os.path.exists(onnx_filepath):
            # convert torch jit model to onnx model
            model = torch.jit.load(filepath).eval()
            dummy_input = torch.randn(bs, 3, 224, 224)
            # Export the model to ONNX
            torch.onnx.export(model, dummy_input, onnx_filepath, export_params=True, opset_version=11)
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        export(onnx_filepath, mindir_filepath, file_format="MINDIR")
    #
    graph = load(mindir_filepath)
    model = nn.GraphCell(graph)
    model.set_train(False)
    for param in model.get_parameters():
        param.requires_grad = False

    return model


def get_feats(videos, detector, bs=10):
    # videos : torch.tensor BCTHW [0, 1]
    detector_kwargs = dict(
        rescale=False, resize=False, return_features=True
    )  # Return raw features before the softmax layer.
    feats = np.empty((0, 400))

    for i in range((len(videos) - 1) // bs + 1):
        feats = np.vstack(
            [
                feats,
                detector(
                    mint.stack([preprocess_single(video) for video in videos[i * bs : (i + 1) * bs]]), **detector_kwargs
                ).asnumpy(),
            ]
        )
    return feats


def get_fvd_feats(videos, i3d, bs=10):
    # videos in [0, 1] as torch tensor BCTHW
    # videos = [preprocess_single(video) for video in videos]
    embeddings = get_feats(videos, i3d, bs)
    return embeddings


def preprocess_single(video, resolution=224, sequence_length=None):
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = ops.interpolate(video, size=target_size, mode="bilinear", align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start : h_start + resolution, w_start : w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2

    return video


"""
Copy-pasted from https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""
from typing import Tuple

import numpy as np
from scipy.linalg import sqrtm


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]
    return mu, sigma


def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    if feats_fake.shape[0] > 1:
        s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    else:
        fid = np.real(m)
    return float(fid)
