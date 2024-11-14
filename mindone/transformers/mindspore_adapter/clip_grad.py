import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F


_clip_grad_value = ops.MultitypeFuncGraph('_clip_grad_value')


@_clip_grad_value.register("Number", "Number", "Tensor")
def __clip_grad_value(max_value, grad):
    """
    Clip gradients.

    Inputs:
        max_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor]: clipped gradients.
    """
    new_grad = C.clip_by_value(
        grad, -max_value, max_value
    )
    return new_grad


_apply_global_norm = ops.MultitypeFuncGraph('_apply_global_norm')


@_apply_global_norm.register("Number", "Tensor", "Tensor")
def __apply_global_norm(clip_coef, x):
    x_dtype = F.dtype(x)
    x = x * clip_coef
    x = F.cast(x, x_dtype)
    return x


_square = ops.MultitypeFuncGraph('_square')


@_square.register("Tensor")
def __square(x):
    return ops.square(x)


_square_sum = ops.MultitypeFuncGraph('_square_sum')


@_square_sum.register("Tensor")
def __square_sum(x):
    return ops.square(x.astype(ms.float32)).sum()


_square_sum_and_all_reduce = ops.MultitypeFuncGraph('_square_sum_and_all_reduce')


@_square_sum_and_all_reduce.register("Tensor")
def __square_sum_and_all_reduce(all_reduce_op, x):
    square_x_sum = ops.square(x.astype(ms.float32)).sum()
    square_x_sum = all_reduce_op(square_x_sum)
    return square_x_sum


@_square_sum.register("Tensor")
def __square_sum(x):
    return ops.square(x.astype(ms.float32)).sum()


hyper_map_op = ops.HyperMap()


def _clip_grad_l2norm(max_norm, grads):
    grads_square_sum = hyper_map_op(_square_sum, grads)
    total_norm = ops.sqrt(ops.addn(grads_square_sum))

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = ops.clamp(clip_coef, None, 1.0)

    clipped_grads = hyper_map_op(F.partial(_apply_global_norm, clip_coef_clamped), grads)
    return clipped_grads


def _clip_grad_l2norm_for_zero(max_norm, all_reduce_op, part_grads):

    grads_square_sum = hyper_map_op(F.partial(_square_sum_and_all_reduce, all_reduce_op), part_grads)
    total_norm = ops.sqrt(ops.addn(grads_square_sum))

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = ops.ones((), dtype=ms.float32) * clip_coef  # necessary on MindSpore 2.3.1 to enable `clip_coef` as a Tensor

    clip_coef_clamped = ops.clamp(clip_coef, None, 1.0)

    clipped_part_grads = hyper_map_op(F.partial(_apply_global_norm, clip_coef_clamped), part_grads)
    return clipped_part_grads


def clip_grad_value(grads, max_value):
    clipped_grads = hyper_map_op(F.partial(_clip_grad_value, max_value), grads)
    return clipped_grads


def clip_grad_norm(grads, max_norm):
    return _clip_grad_l2norm(max_norm, grads)


def clip_grad_norm_for_zero(part_grads, max_norm, all_reduce_op):
    return _clip_grad_l2norm_for_zero(max_norm, all_reduce_op, part_grads)
