import os
from typing import Dict, Union

import numpy as np
import safetensors.numpy

from mindspore import Parameter, Tensor


def load_safetensors(filename: Union[str, os.PathLike], force_fp32: bool = False) -> Dict[str, Tensor]:
    flat = safetensors.numpy.load_file(filename)
    output = _np2ms(flat, force_fp32=force_fp32)
    return output


def _np2ms(np_dict: Dict[str, np.ndarray], force_fp32: bool = False) -> Dict[str, Tensor]:
    for k, v in np_dict.items():
        if force_fp32 and v.dtype == np.float16:
            v = v.astype(np.float32)
        np_dict[k] = Parameter(v, name=k)
    return np_dict
