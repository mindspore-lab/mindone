import numpy as np

import mindspore as ms
from mindspore import ParallelMode

_DTYPE_2_STRING = {
    ms.float16: "float16",
    ms.bfloat16: "bfloat16",
    ms.float32: "float32",
    ms.float64: "float64",
    ms.uint8: "uint8",
    ms.int8: "int8",
    ms.int16: "int16",
    ms.int32: "int32",
    ms.int64: "int64",
    ms.bool_: "bool",
}

_STRING_2_DTYPE = {
    "float16": ms.float16,
    "bfloat16": ms.bfloat16,
    "float32": ms.float32,
    "float64": ms.float64,
    "uint8": ms.uint8,
    "int8": ms.int8,
    "int16": ms.int16,
    "int32": ms.int32,
    "int64": ms.int64,
    "bool": ms.bool_,
}

_MIN_INT8 = ms.tensor(np.iinfo(np.int8).min, dtype=ms.int8)
_MIN_INT16 = ms.tensor(np.iinfo(np.int16).min, dtype=ms.int16)
_MIN_INT32 = ms.tensor(np.iinfo(np.int32).min, dtype=ms.int32)
_MIN_INT64 = ms.tensor(np.iinfo(np.int64).min, dtype=ms.int64)
_MIN_FP16 = ms.tensor(np.finfo(np.float16).min, dtype=ms.float16)
_MIN_FP32 = ms.tensor(np.finfo(np.float32).min, dtype=ms.float32)
_MIN_FP64 = ms.tensor(np.finfo(np.float64).min, dtype=ms.float64)
_MIN_BF16 = ms.tensor(float.fromhex("-0x1.fe00000000000p+127"), dtype=ms.bfloat16)
_MAX_FP16 = ms.tensor(np.finfo(np.float16).max, dtype=ms.float16)
_MAX_FP32 = ms.tensor(np.finfo(np.float32).max, dtype=ms.float32)
_MAX_FP64 = ms.tensor(np.finfo(np.float64).max, dtype=ms.float64)
_MAX_BF16 = ms.tensor(float.fromhex("0x1.fe00000000000p+127"), dtype=ms.bfloat16)


_DTYPE_2_MIN = {
    ms.int8: _MIN_INT8,
    ms.int16: _MIN_INT16,
    ms.int32: _MIN_INT32,
    ms.int64: _MIN_INT64,
    ms.float16: _MIN_FP16,
    ms.float32: _MIN_FP32,
    ms.float64: _MIN_FP64,
    ms.bfloat16: _MIN_BF16,
}

_DTYPE_2_MAX = {
    ms.float16: _MAX_FP16,
    ms.float32: _MAX_FP32,
    ms.float64: _MAX_FP64,
    ms.bfloat16: _MAX_BF16,
}

TORCH_TO_MINDSPORE_DTYPE_MAP = {
    "torch.float32": ms.float32,
    "torch.bfloat16": ms.bfloat16,
    "torch.float16": ms.float16,
}


def dtype_to_min(dtype):
    return _DTYPE_2_MIN.get(dtype, "others dtype")


def dtype_to_max(dtype):
    return _DTYPE_2_MAX.get(dtype, "others dtype")


def dtype_to_str(dtype):
    return _DTYPE_2_STRING.get(dtype, "others dtype")


def str_to_dtype(dtype):
    return _STRING_2_DTYPE.get(dtype, "others dtype")


def _is_parallel():
    return ms.context.get_auto_parallel_context("parallel_mode") not in (ParallelMode.STAND_ALONE,)


def _is_graph():
    return ms.context.get_context("mode") == ms.GRAPH_MODE


def _is_ascend():
    return ms.context.get_context("device_target") == "Ascend"


# FIXME: Can't work on MindSpore 2.3.0
# @ms.constexpr(reuse_result=False)
# def _tensor_2_tuple(input):
#     return tuple(input.asnumpy().tolist())


# equivalent implementation of torch Tensor.unfold
def unfold(tensor, dimension, size, step=1):
    dimension = dimension if dimension >= 0 else tensor.ndim + dimension
    target_dim_size = tensor.shape[dimension]
    window_count = (target_dim_size - size) // step + 1
    windows = []
    for i in range(window_count):
        start_idx = i * step
        end_idx = start_idx + size

        slices = [slice(None)] * tensor.ndim
        slices[dimension] = slice(start_idx, end_idx)

        window = tensor[tuple(slices)]
        windows.append(window)

    result = ms.mint.stack(windows, dim=dimension + 1)
    result = result.movedim(dimension, -1)

    return result
