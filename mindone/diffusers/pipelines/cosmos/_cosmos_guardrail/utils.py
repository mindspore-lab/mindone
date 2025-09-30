import logging
import os
import tempfile
from typing import Callable, Generic, List, Optional, Tuple, TypeVar
from urllib.parse import unquote, urlparse

import imageio
import PIL.Image
import requests

from mindspore import Tensor

T_co = TypeVar("T_co", covariant=True)


def get_logger(name):
    """Simple logger function"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_video(
    video: str,
    convert_method: Optional[Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]]] = None,
) -> List[PIL.Image.Image]:
    """
    Loads `video` to a list of PIL Image.

    Args:
        video (`str`):
            A URL or Path to a video to convert to a list of PIL Image format.
        convert_method (Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]], *optional*):
            A conversion method to apply to the video after loading it.

    Returns:
        `List[PIL.Image.Image]`:
            The video as a list of PIL images.
    """
    is_url = video.startswith("http://") or video.startswith("https://")
    is_file = os.path.isfile(video)
    was_tempfile_created = False

    if not (is_url or is_file):
        raise ValueError(
            f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {video} is not a valid path."
        )

    if is_url:
        response = requests.get(video, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Failed to download video. Status code: {response.status_code}")

        parsed_url = urlparse(video)
        file_name = os.path.basename(unquote(parsed_url.path))

        suffix = os.path.splitext(file_name)[1] or ".mp4"
        video_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name

        was_tempfile_created = True

        video_data = response.iter_content(chunk_size=8192)
        with open(video_path, "wb") as f:
            for chunk in video_data:
                f.write(chunk)

        video = video_path

    pil_images = []
    if video.endswith(".gif"):
        gif = PIL.Image.open(video)
        try:
            while True:
                pil_images.append(gif.copy())
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
    else:
        # Try using imageio directly without checking availability
        try:
            with imageio.get_reader(video) as reader:
                # Read all frames
                for frame in reader:
                    pil_images.append(PIL.Image.fromarray(frame))
        except Exception as e:
            raise RuntimeError(f"Error loading video with imageio: {e}")

    if was_tempfile_created:
        os.remove(video_path)

    if convert_method is not None:
        pil_images = convert_method(pil_images)

    return pil_images


# copy/adpated from torch.utils.data.Dataset
class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`."""

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")


# copy/adpated from torch.utils.data.TensorDataset
class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]
