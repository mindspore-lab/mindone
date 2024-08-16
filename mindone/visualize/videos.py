import os
import subprocess as sp
from typing import Union

import imageio
import numpy as np

__all__ = ["save_videos", "create_video_from_numpy_frames"]


def create_video_from_rgb_numpy_arrays(image_arrays, output_file, fps: Union[int, float] = 30):
    """
    Creates an MP4 video file from a series of RGB NumPy array images.

    Parameters:
    image_arrays (list): A list of RGB NumPy array images.
    output_file (str): The path and filename of the output MP4 video file.
    fps (int, float): The desired frames per second for the output video. Default is 30.
    """
    # Get the dimensions of the first image
    height, width, _ = image_arrays[0].shape

    command = f"ffmpeg -y -f rawvideo -vcodec rawvideo \
                -s {width}:{height} -pix_fmt rgb24 -r {fps} \
                -i - -an -vcodec libx264 -pix_fmt yuv420p \
                -b:v 1024k -loglevel quiet {output_file}"

    pipe = sp.Popen(command, stdin=sp.PIPE)
    for frame in image_arrays:
        pipe.stdin.write(frame.astype(np.uint8).tobytes())
    pipe.stdin.close()
    pipe.wait()


def create_video_from_numpy_frames(frames: np.ndarray, path: str, fps: Union[int, float] = 8, fmt="gif", loop=0):
    """
    Args:
        frames: shape (f h w 3), range [0, 255], order rgb
    """
    if fmt == "gif":
        imageio.mimsave(path, frames, duration=1 / fps, loop=loop)
    if fmt == "png":
        for i in range(len(frames)):
            imageio.imwrite(path.replace(".png", f"-{i:04}.png"), frames[i])
    elif fmt == "mp4":
        create_video_from_rgb_numpy_arrays(frames, path, fps=fps)


def save_videos(frames: np.ndarray, path: str, fps: Union[int, float] = 8, loop=0, concat=False):
    """
    Save video frames to gif or mp4 files
    Args:
        frames: video frames in shape (b f h w 3), pixel value in [0, 1], RGB mode.
        path:  file path to save the output gif
        fps: frames per sencond in the output gif. 1/fps = display duration per frame
        concat: if True and b>1, all videos will be concatnated in grids and saved as one gif.
        loop: number of loops to play. If 0, it will play endlessly.
    """
    fmt = path.split(".")[-1]
    assert fmt in ["gif", "mp4", "png"]

    # input frames: (b f H W 3), normalized to [0, 1]
    frames = (frames * 255).round().clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if len(frames.shape) == 4:
        create_video_from_numpy_frames(frames, path, fps, fmt, loop)
    else:
        b, f, h, w, _ = frames.shape
        if b > 1:
            if concat:
                canvas = np.array((f, h, w * b, 3), dtype=np.uint8)
                for idx in range(b):
                    canvas[:, :, (w * idx) : (w * (idx + 1)), :] = frames[idx]
                create_video_from_numpy_frames(canvas, path, fps, fmt, loop)
            else:
                for idx in range(b):
                    cur_path = path.replace(f".{fmt}", f"-{idx}.{fmt}")
                    create_video_from_numpy_frames(frames[idx], cur_path, fps, fmt, loop)
        else:
            create_video_from_numpy_frames(frames[0], path, fps, fmt, loop)
