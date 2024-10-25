import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F


_clip_grad = ops.MultitypeFuncGraph('_clip_grad')


@_clip_grad.register("Number", "Number", "Tensor")
def __clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor]: clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    if clip_type == 0:
        new_grad = C.clip_by_value(
            grad, -clip_value, clip_value
        )
    else:
        new_grad = ops.clip_by_norm(grad, clip_value)
    return new_grad


_apply_global_norm = ops.MultitypeFuncGraph('_apply_global_norm')


@_apply_global_norm.register("Number", "Tensor", "Tensor")
def __apply_global_norm(clip_norm, global_norm, x):
    x_dtype = F.dtype(x)
    x = x * clip_norm / global_norm
    x = F.cast(x, x_dtype)
    return x


_l2_norm = ops.MultitypeFuncGraph('_l2_norm')


@_l2_norm.register("Tensor")
def __l2_norm(x):
    axis = ()
    for i in range(x.dim()):
        axis += (i,)
    norm = ops.LpNorm(axis)(x)
    return norm


_square = ops.MultitypeFuncGraph('_square')


@_square.register("Tensor")
def __square(x):
    return ops.square(x)


class L2Norm(nn.Cell):
    def __init__(self):
        super().__init__()
        self.l2_norm_1 = ops.LpNorm((0,))
        self.l2_norm_2 = ops.LpNorm((0, 1))
        self.l2_norm_3 = ops.LpNorm((0, 1, 2))
        self.l2_norm_4 = ops.LpNorm((0, 1, 2, 3))

    def construct(self, x):
        if x.ndim == 1:
            norm = self.l2_norm_1(x)
        elif x.ndim == 2:
            norm = self.l2_norm_2(x)
        elif x.ndim == 3:
            norm = self.l2_norm_3(x)
        else:
            norm = self.l2_norm_4(x)
        return norm


class _ClipByGlobalNormFix(nn.Cell):
    def __init__(self, clip_norm=1.0):
        super().__init__()
        self.clip_norm = Tensor([clip_norm], ms.float32)
        self.hyper_map = ops.HyperMap()
        self.greater_equal = ops.GreaterEqual()
        self.l2norm = L2Norm()

    def construct(self, x):
        norms = self.hyper_map(self.l2norm, x)
        norms_square = self.hyper_map(ops.square, norms)
        global_norm = ops.sqrt(ops.addn(norms_square)).astype(ms.float32)

        cond = self.greater_equal(global_norm, self.clip_norm)
        global_norm = F.select(cond, global_norm, self.clip_norm)
        clip_x = self.hyper_map(F.partial(_apply_global_norm, self.clip_norm, global_norm), x)
        return clip_x


hyper_map_op = ops.HyperMap()


def _clip_grad_global_fix(clip_norm, grads):
    # norms = hyper_map_op(_l2_norm, grads)
    # norms_square = hyper_map_op(_square, norms)
    # global_norm = ops.sqrt(ops.addn(norms_square)).astype(ms.float32)

    global_norm = ops.ones((), ms.float32)

    cond = ops.greater_equal(global_norm, clip_norm)
    global_norm = F.select(cond, global_norm, clip_norm)
    clip_grads = hyper_map_op(F.partial(_apply_global_norm, clip_norm, global_norm), grads)
    return clip_grads


def clip_grad(grads, clip_norm):
    clip_value = hyper_map_op(F.partial(_clip_grad, 1, clip_norm), grads)
    return clip_value


def clip_grad_global(grads, clip_norm):
    clip_value = _clip_grad_global_fix(clip_norm, grads)
    return clip_value
