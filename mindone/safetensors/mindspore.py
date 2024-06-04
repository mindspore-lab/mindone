import os
from typing import Dict, Optional, Union

import numpy as np
from safetensors import numpy

import mindspore as ms


def save(tensors: Dict[str, ms.Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, ms.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors_ms import save
    import mindspore.ops as ops

    tensors = {"embedding": ops.zeros((512, 1024)), "attention": ops.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    np_tensors = _ms2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


def save_file(
    tensors: Dict[str, ms.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, ms.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        filename (`str`, or `os.PathLike`)):
            The filename we're saving into.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `None`

    Example:

    ```python
    from safetensors_ms import save_file
    import mindspore.ops as ops

    tensors = {"embedding": ops.zeros((512, 1024)), "attention": ops.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    np_tensors = _ms2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata)


def load(data: bytes) -> Dict[str, ms.Tensor]:
    """
    Loads a safetensors file into mindspore format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, ms.Tensor]`: dictionary that contains name as key, value as `ms.Tensor` on cpu

    Example:

    ```python
    from safetensors_ms import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = numpy.load(data)
    return _np2ms(flat)


def load_file(filename: Union[str, os.PathLike]) -> Dict[str, ms.Tensor]:
    """
    Loads a safetensors file into mindspore format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors

    Returns:
        `Dict[str, ms.Tensor]`: dictionary that contains name as key, value as `ms.Tensor`

    Example:

    ```python
    from safetensors_ms import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    flat = numpy.load_file(filename)
    output = _np2ms(flat)
    return output


def _np2ms(np_dict: Dict[str, np.ndarray]) -> Dict[str, ms.Tensor]:
    for k, v in np_dict.items():
        np_dict[k] = ms.Parameter(v, name=k)
    return np_dict


def _ms2np(ms_dict: Dict[str, ms.Tensor]) -> Dict[str, np.array]:
    for k, v in ms_dict.items():
        ms_dict[k] = v.numpy()
    return ms_dict
