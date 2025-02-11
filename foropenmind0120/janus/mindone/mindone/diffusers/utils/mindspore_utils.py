# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MindSpore utilities: Utilities related to MindSpore
"""

from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import nn, ops

from . import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_MIN_FP16 = ms.tensor(np.finfo(np.float16).min, dtype=ms.float16)
_MIN_FP32 = ms.tensor(np.finfo(np.float32).min, dtype=ms.float32)
_MIN_FP64 = ms.tensor(np.finfo(np.float64).min, dtype=ms.float64)
_MIN_BF16 = ms.tensor(float.fromhex("-0x1.fe00000000000p+127"), dtype=ms.bfloat16)


# Copied from mindone.transformers.modeling_attn_mask_utils.dtype_to_min
def dtype_to_min(dtype):
    if dtype == ms.float16:
        return _MIN_FP16
    if dtype == ms.float32:
        return _MIN_FP32
    if dtype == ms.float64:
        return _MIN_FP64
    if dtype == ms.bfloat16:
        return _MIN_BF16
    else:
        raise ValueError(f"Only support get minimum value of (bfloat16, float16, float32, float64), but got {dtype}")


def get_state_dict(module: nn.Cell, name_prefix="", recurse=True):
    """
    A function attempting to achieve an effect similar to torch's `nn.Module.state_dict()`.

    Due to MindSpore's unique parameter naming mechanism, this function performs operations
    on the prefix of parameter names. This ensures that parameters can be correctly loaded
    using `mindspore.load_param_into_net()` when there are discrepancies between the parameter
    names of the target_model and source_model.
    """
    param_generator = module.parameters_and_names(name_prefix=name_prefix, expand=recurse)

    param_dict = OrderedDict()
    for name, param in param_generator:
        param.name = name
        param_dict[name] = param
    return param_dict


def randn(
    size: Union[Tuple, List], generator: Optional["np.random.Generator"] = None, dtype: Optional["ms.Type"] = None
):
    if generator is None:
        generator = np.random.default_rng()

    return ms.tensor(generator.standard_normal(size), dtype=dtype)


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["np.random.Generator"], "np.random.Generator"]] = None,
    dtype: Optional["ms.Type"] = None,
):
    """A helper function to create random tensors with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually.
    """
    batch_size = shape[0]

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [randn(shape, generator=generator[i], dtype=dtype) for i in range(batch_size)]
        latents = ops.cat(latents, axis=0)
    else:
        latents = randn(shape, generator=generator, dtype=dtype)

    return latents
