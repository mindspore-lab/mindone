import logging
import os

import cv2
import imageio
import numpy as np

__all__ = ["export_to_gif", "export_to_video"]

_logger = logging.getLogger(__name__)


def export_to_gif(frames: np.ndarray, path: str, fps: int = 8, loop: int = 0, concat: bool = False) -> None:
    """
    Export video frames to a gif file.

    Args:
        frames: [batch of] video frames in shape ([b] f h w 3) and RGB mode.
        path:  file path to save the output gif(s).
        fps: frames per second in the output gif. 1/fps = display duration per frame
        concat: if True and b>1, all videos will be concatenated horizontally and saved as one.
        loop: number of loops to play. If 0, it will play endlessly.
    """
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)

    duration = 1 / fps

    frames = frames.squeeze()  # squeeze batch dimension if equal to 1 for simplicity
    if len(frames.shape) == 4:
        imageio.mimsave(path, frames, duration=duration, loop=loop)
        _logger.info(f"Generated GIF saved to {path}")
    else:
        if concat:
            frames = np.concatenate(frames, axis=2)
            imageio.mimsave(path, frames, duration=duration, loop=loop)
            _logger.info(f"Generated GIF saved to {path}")
        else:
            for idx in range(len(frames)):
                new_path = path.replace(".gif", f"-{idx}.gif")
                imageio.mimsave(new_path, frames[idx], duration=duration, loop=loop)
                _logger.info(f"Generated GIF saved to {new_path}")


def _write_video(frames: np.ndarray, path: str, fps: int = 25, codec: str = "mp4v") -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    _logger.info(f"Generated video saved to {path}")


def export_to_video(frames: np.ndarray, path: str, fps: int = 25, concat: bool = False, codec: str = "mp4v") -> None:
    """
    Export video frames to a video file.

    Args:
        frames: [Batch of] video frames in shape ([b] f h w 3) and RGB mode.
        path: Path to the output video file.
        fps: Frames per second of the output video. Default is 25.
        concat: If True, all videos will be concatenated horizontally and saved as one.
        codec: Video codec to be used for encoding. Default is "mp4v".
    """
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)

    frames = frames.squeeze()  # squeeze batch dimension if equal to 1 for simplicity
    if len(frames.shape) == 4:
        _write_video(frames, path, fps, codec)
    else:
        if concat:
            frames = np.concatenate(frames, axis=2)
            _write_video(frames, path, fps, codec)
        else:
            for i in range(len(frames)):
                name, ext = path.rsplit(".", 1)
                new_path = name + f"-{i}." + ext
                _write_video(frames[i], new_path, fps, codec)
