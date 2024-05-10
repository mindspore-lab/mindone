import os

import imageio
import numpy as np


def save_videos(videos: np.ndarray, path: str, fps=8, concat=False):
    # videos: (b f H W 3), normalized to [0, 1]
    videos = (videos * 255).round().clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if len(videos.shape) == 4:
        imageio.mimsave(path, videos, fps=fps)
    else:
        b, f, h, w, _ = videos.shape
        if b > 1:
            if concat:
                canvas = np.array((f, h, w * b, 3), dtype=np.uint8)
                for idx in range(b):
                    canvas[:, :, (w * idx) : (w * (idx + 1)), :] = videos[idx]
                imageio.mimsave(path, canvas, fps=fps)
            else:
                for idx in range(b):
                    # concat in Width
                    imageio.mimsave(path.replace(".gif", f"-{idx}.gif"), videos[idx], fps=fps)
        else:
            imageio.mimsave(path, videos[0], fps=fps)


def video_tensor_to_gif(images: np.ndarray, path: str, duration: int = 120, save_frames: bool = False) -> None:
    """images: f x h x w x c"""
    images = (images * 255).round().clip(0, 255).astype(np.uint8)
    images = [x for x in images]
    imageio.mimwrite(path, images, duration=duration)
    if save_frames:
        root = os.path.join(os.path.splitext(path)[0], "frames")
        if not os.path.isdir(root):
            os.makedirs(root)
        for i, x in enumerate(images):
            imageio.imwrite(os.path.join(root, f"{i:04d}.jpg"), x)
