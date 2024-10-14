import mindspore as ms
from mindspore import mint, ops
from mindspore.ops.function.array_func import chunk_ext, repeat_interleave_ext

use_dynamic_ops = False


# make sure it's invoked before model init to switch mint/ops module
def set_dynamic_mode(dynamic: bool = True):
    global use_dynamic_ops
    use_dynamic_ops = dynamic


def check_dynamic_mode():
    return use_dynamic_ops


def repeat_interleave_ext_v2(input, repeats, axis=None):
    # A more efficient implementation for replacing mint.repeat_interleave_ext
    if isinstance(repeats, ms.Tensor):
        if repeats.ndim > 1:
            raise ValueError(f"repeats must be int, but get Tensor and ndim > 1, repeats.ndim {repeats.ndim}")
        else:
            repeats = int(repeats)
    if isinstance(repeats, (tuple, list)):
        if len(repeats) > 1:
            raise ValueError(f"repeats must be int, but get list and len > 1, len(repeats) {len(repeats)}")
        else:
            repeats = repeats[0]
    if not isinstance(repeats, int):
        raise ValueError(f"repeats must be int, but get {repeats}")
    if axis is None:
        input = input.reshape[-1]
        axis = 0

    if not isinstance(axis, int):
        raise ValueError(f"axis must be int, but get {axis}")
    axis = axis + input.ndim if axis < 0 else axis
    x_shape = input.shape
    tile_axis = [1]
    y_shape = list(x_shape)
    y_shape[axis] = -1
    for i in range(1, input.ndim + 1):
        if i == axis + 1:
            tile_axis.append(repeats)
        else:
            tile_axis.append(1)
    input = ops.expand_dims(input, axis + 1)

    return mint.tile(input, tuple(tile_axis)).reshape(y_shape)


def get_repeat_interleave_op():
    mode = ms.get_context("mode")
    if (mode == 0) and (not check_dynamic_mode()):
        # provide better performance for static shape in graph mode
        return ops.repeat_interleave
    else:
        # FIXME: check overflow for v2
        # return repeat_interleave_ext_v2
        return repeat_interleave_ext


def get_chunk_op():
    mode = ms.get_context("mode")
    if (mode == 0) and (not check_dynamic_mode()):
        return ops.chunk
    else:
        return chunk_ext


def get_split_op():
    mode = ms.get_context("mode")
    if (mode == 0) and (not check_dynamic_mode()):
        return ops.split
    else:
        return mint.split
