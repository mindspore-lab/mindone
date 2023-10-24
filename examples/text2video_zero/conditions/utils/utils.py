import os

import cv2
import imageio
import numpy as np
from conditions.canny.canny_detector import CannyDetector
from einops import rearrange

import mindspore as ms
from mindspore.dataset.vision import Inter, Resize

annotator_ckpts_path = os.path.join(os.path.dirname(__file__), "ckpts")


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(image: np.ndarray, resolution: int) -> np.ndarray:
    h, w = image.shape[:2]
    k = resolution / min(h, w)
    h = int(np.round(h * k / 64.0)) * 64
    w = int(np.round(w * k / 64.0)) * 64
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)


def prepare_video(video_path: str, resolution: int, normalize=True, output_fps: int = -1):
    vr = cv2.VideoCapture(video_path)
    initial_fps = vr.get(cv2.CAP_PROP_FPS)
    if output_fps == -1:
        output_fps = int(initial_fps)

    assert output_fps > 0

    frame_list = []
    while True:
        ret, frame = vr.read()
        if not ret:
            break
        frame_list.append(frame)

    video = np.stack(frame_list, axis=0)
    _, h, w, _ = video.shape

    # Use max if you want the larger side to be equal to resolution (e.g. 512)
    k = float(resolution) / max(h, w)
    h *= k
    w *= k
    h = int(np.round(h / 64.0)) * 64
    w = int(np.round(w / 64.0)) * 64
    video = Resize((h, w), Inter.BILINEAR)(video)
    if normalize:
        video = video / 127.5 - 1.0
    return video, output_fps


def pre_process_canny(input_video, low_threshold=100, high_threshold=200):
    detected_maps = []
    apply_canny = CannyDetector()
    for frame in input_video:
        frame = frame.astype(np.uint8)
        detected_map = apply_canny(frame, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)
    control = rearrange(detected_maps, "f h w c -> f c h w")
    control = ms.Tensor(control.copy() / 255.0).float()
    return control


def create_video(frames, fps, mode="mp4", rescale=False, path=None):
    if path is None:
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, "movie.mp4")

    path_prefix = path.split(".")[0]
    path = path_prefix + "." + mode

    outputs = []
    for i, x in enumerate(frames):
        if rescale:
            x = (x + 1.0) / 2.0
        x = (x * 255).astype(np.uint8)

        outputs.append(x)

    imageio.mimsave(path, outputs, fps=fps)
    return path
