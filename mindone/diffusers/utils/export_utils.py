"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/utils/export_utils.py."""

import io
import random
import struct
import tempfile
from contextlib import contextmanager
from typing import List, Optional, Union

import numpy as np
import PIL.Image
import PIL.ImageOps

from .import_utils import BACKENDS_MAPPING, is_imageio_available, is_opencv_available
from .logging import get_logger

global_rng = random.Random()

logger = get_logger(__name__)


@contextmanager
def buffered_writer(raw_f):
    f = io.BufferedWriter(raw_f)
    yield f
    f.flush()


def export_to_gif(image: List[PIL.Image.Image], output_gif_path: str = None, fps: int = 10) -> str:
    if output_gif_path is None:
        output_gif_path = tempfile.NamedTemporaryFile(suffix=".gif").name

    image[0].save(
        output_gif_path,
        save_all=True,
        append_images=image[1:],
        optimize=False,
        duration=1000 // fps,
        loop=0,
    )
    return output_gif_path


def export_to_ply(mesh, output_ply_path: str = None):
    """
    Write a PLY file for a mesh.
    """
    if output_ply_path is None:
        output_ply_path = tempfile.NamedTemporaryFile(suffix=".ply").name

    coords = mesh.verts.numpy()
    faces = mesh.faces.numpy()
    rgb = np.stack([mesh.vertex_channels[x].numpy() for x in "RGB"], axis=1)

    with buffered_writer(open(output_ply_path, "wb")) as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(bytes(f"element vertex {len(coords)}\n", "ascii"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if rgb is not None:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
        if faces is not None:
            f.write(bytes(f"element face {len(faces)}\n", "ascii"))
            f.write(b"property list uchar int vertex_index\n")
        f.write(b"end_header\n")

        if rgb is not None:
            rgb = (rgb * 255.499).round().astype(int)
            vertices = [
                (*coord, *rgb)
                for coord, rgb in zip(
                    coords.tolist(),
                    rgb.tolist(),
                )
            ]
            format = struct.Struct("<3f3B")
            for item in vertices:
                f.write(format.pack(*item))
        else:
            format = struct.Struct("<3f")
            for vertex in coords.tolist():
                f.write(format.pack(*vertex))

        if faces is not None:
            format = struct.Struct("<B3I")
            for tri in faces.tolist():
                f.write(format.pack(len(tri), *tri))

    return output_ply_path


def export_to_obj(mesh, output_obj_path: str = None):
    if output_obj_path is None:
        output_obj_path = tempfile.NamedTemporaryFile(suffix=".obj").name

    verts = mesh.verts.numpy()
    faces = mesh.faces.numpy()

    vertex_colors = np.stack([mesh.vertex_channels[x].numpy() for x in "RGB"], axis=1)
    vertices = [
        "{} {} {} {} {} {}".format(*coord, *color) for coord, color in zip(verts.tolist(), vertex_colors.tolist())
    ]

    faces = ["f {} {} {}".format(str(tri[0] + 1), str(tri[1] + 1), str(tri[2] + 1)) for tri in faces.tolist()]

    combined_data = ["v " + vertex for vertex in vertices] + faces

    with open(output_obj_path, "w") as f:
        f.writelines("\n".join(combined_data))


def _legacy_export_to_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 10
):
    if is_opencv_available():
        import cv2
    else:
        raise ImportError(BACKENDS_MAPPING["opencv"][1].format("export_to_video"))
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

    return output_video_path


def export_to_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str = None,
    fps: int = 10,
    quality: float = 5.0,
    bitrate: Optional[int] = None,
    macro_block_size: Optional[int] = 16,
) -> str:
    """
    quality:
        Video output quality. Default is 5. Uses variable bit rate. Highest quality is 10, lowest is 0. Set to None to
        prevent variable bitrate flags to FFMPEG so you can manually specify them using output_params instead.
        Specifying a fixed bitrate using `bitrate` disables this parameter.

    bitrate:
        Set a constant bitrate for the video encoding. Default is None causing `quality` parameter to be used instead.
        Better quality videos with smaller file sizes will result from using the `quality` variable bitrate parameter
        rather than specifying a fixed bitrate with this parameter.

    macro_block_size:
        Size constraint for video. Width and height, must be divisible by this number. If not divisible by this number
        imageio will tell ffmpeg to scale the image up to the next closest size divisible by this number. Most codecs
        are compatible with a macroblock size of 16 (default), some can go smaller (4, 8). To disable this automatic
        feature set it to None or 1, however be warned many players can't decode videos that are odd in size and some
        codecs will produce poor results or fail. See https://en.wikipedia.org/wiki/Macroblock.
    """
    # TODO: Dhruv. Remove by Diffusers release 0.33.0
    # Added to prevent breaking existing code
    if not is_imageio_available():
        logger.warning(
            (
                "It is recommended to use `export_to_video` with `imageio` and `imageio-ffmpeg` as a backend. \n"
                "These libraries are not present in your environment. Attempting to use legacy OpenCV backend to export video. \n"
                "Support for the OpenCV backend will be deprecated in a future Diffusers version"
            )
        )
        return _legacy_export_to_video(video_frames, output_video_path, fps)

    if is_imageio_available():
        import imageio
    else:
        raise ImportError(BACKENDS_MAPPING["imageio"][1].format("export_to_video"))

    try:
        imageio.plugins.ffmpeg.get_exe()
    except AttributeError:
        raise AttributeError(
            (
                "Found an existing imageio backend in your environment. Attempting to export video with imageio. \n"
                "Unable to find a compatible ffmpeg installation in your environment to use with imageio. Please install via `pip install imageio-ffmpeg"
            )
        )

    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    with imageio.get_writer(
        output_video_path, fps=fps, quality=quality, bitrate=bitrate, macro_block_size=macro_block_size
    ) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    return output_video_path
