import mindspore as ms
import numpy as np
from mindspore import Tensor, Parameter, context, ParallelMode


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
    ms.bool_:  "bool",
}

_MIN_FP16 = ms.tensor(np.finfo(np.float16).min, dtype=ms.float16)
_MIN_FP32 = ms.tensor(np.finfo(np.float32).min, dtype=ms.float32)
_MIN_FP64 = ms.tensor(np.finfo(np.float64).min, dtype=ms.float64)
_MIN_BF16 = ms.tensor(float.fromhex("-0x1.fe00000000000p+127"), dtype=ms.bfloat16)


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
        raise ValueError(f"Only support get minimum value of (float16, ), but got {dtype}")


def _is_parallel():
    return context.get_auto_parallel_context("parallel_mode") not in (ParallelMode.STAND_ALONE,)


def _is_graph():
    return context.get_context("mode") == context.GRAPH_MODE


# FIXME: Can't work on MindSpore 2.3.0
# @ms.constexpr(reuse_result=False)
# def _tensor_2_tuple(input):
#     return tuple(input.asnumpy().tolist())
