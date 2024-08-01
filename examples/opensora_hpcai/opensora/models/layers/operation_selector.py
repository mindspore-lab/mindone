import mindspore as ms
from mindspore import mint, ops
from mindspore.ops.function.array_func import repeat_interleave_ext

use_dynamic_ops = False


# make sure it's invoked before model init to switch mint/ops module
def set_dynamic_mode(dynamic: bool = True):
    global use_dynamic_ops
    use_dynamic_ops = dynamic


def check_dynamic_mode():
    return use_dynamic_ops


def get_repeat_interleave_op():
    mode = ms.get_context("mode")
    if (mode == 0) and (not check_dynamic_mode()):
        # provide better performance for static shape in graph mode
        return ops.repeat_interleave
    else:
        return repeat_interleave_ext


def get_chunk_op():
    mode = ms.get_context("mode")
    if (mode == 0) and (not check_dynamic_mode()):
        return ops.chunk
    else:
        return mint.chunk


def get_split_op():
    mode = ms.get_context("mode")
    if (mode == 0) and (not check_dynamic_mode()):
        return ops.split
    else:
        return mint.split
