import os

import imageio
import numpy as np


def save_videos(frames: np.ndarray, path: str, fps=8, concat=False):
    """
    Save video frames to gif files
    Args:
        frames: video frames in shape (b f h w 3), pixel value in [0, 1], RGB mode.
        path:  file path to save the output gif
        fps: frames per sencond in the output gif. 1/fps = display duration per frame
        concat: if True and b>1, all videos will be concatnated in grids and saved as one gif.
    """
    # input frames: (b f H W 3), normalized to [0, 1]
    frames = (frames * 255).round().clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    duration = 1 / fps
    if len(frames.shape) == 4:
        imageio.mimsave(path, frames, duration=duration)
    else:
        b, f, h, w, _ = frames.shape
        if b > 1:
            if concat:
                canvas = np.array((f, h, w * b, 3), dtype=np.uint8)
                for idx in range(b):
                    canvas[:, :, (w * idx) : (w * (idx + 1)), :] = frames[idx]
                imageio.mimsave(path, canvas, duration=duration)
            else:
                for idx in range(b):
                    # concat in Width dimension
                    imageio.mimsave(path.replace(".gif", f"-{idx}.gif"), frames[idx], fps=fps)
        else:
            imageio.mimsave(path, frames[0], duration=duration)
