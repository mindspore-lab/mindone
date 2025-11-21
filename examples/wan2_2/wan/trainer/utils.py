import os
from functools import partial
from typing import Tuple

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.distributed as dist


def map_(*args, **kwargs) -> Tuple:
    """
    same as `map`, but returns a tuple instead of a iterator.
    This is useful for when operation is done in place, to make sure the operation is applied to all elements.
    """
    return tuple(map(*args, **kwargs))


def syn_gradients(gradients: Tuple[ms.Tensor, ...]) -> None:
    """
    Synchronize gradients across all devices.
    """
    if not dist.is_initialized():
        return

    size = dist.get_world_size()
    if size == 1:
        return

    map_(dist.all_reduce, gradients)
    map_(lambda x: x.mul_(1 / size), gradients)


def save_checkpoint(trainable_parameters: ms.ParameterTuple, outdir: str) -> None:
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    ms.save_checkpoint(list(trainable_parameters), os.path.join(outdir, "model.ckpt"))


def clip_by_global_norm(grads: Tuple[ms.Tensor, ...], max_norm: float, norm_type: float = 2.0) -> ms.Tensor:
    """
    Clips the gradients by global norm in place.
    Returns the total norm of the gradients.
    """
    total_norm = _get_total_norm(grads, norm_type)
    _clip_grads_with_norm_(grads, max_norm, total_norm)
    return total_norm


def _get_total_norm(tensors: Tuple[ms.Tensor, ...], norm_type: float = 2.0) -> ms.Tensor:
    norms = map_(partial(mint.linalg.vector_norm, ord=norm_type), tensors)
    total_norm = mint.linalg.vector_norm(mint.stack(norms), ord=norm_type)
    return total_norm


def _clip_grads_with_norm_(grads: Tuple[ms.Tensor, ...], max_norm: float, total_norm: ms.Tensor) -> None:
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = mint.clamp(clip_coef, max=1.0)
    map_(lambda g: g.mul_(clip_coef_clamped), grads)
