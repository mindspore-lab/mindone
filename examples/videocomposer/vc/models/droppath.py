import mindspore as ms
from mindspore import nn, ops
from mindspore.ops import ones
from .attention import (
    BasicTransformerBlock,
    CrossAttention,
    FeedForward,
    GroupNorm,
    RelativePositionBias,
    SpatialTransformer,
    TemporalAttentionBlock,
    TemporalAttentionMultiBlock,
    TemporalConvBlock_v2,
    TemporalTransformer,
    default,
    is_old_ms_version,
    zero_module,
)


def gen_zero_keep_mask(dist, bs):
    # dist: [p_zero, p_keep, 1-(p_zero, p_keep)] 
    #dist = ms.Tensor([p_zero, p_keep, 1-(p_zero+p_keep)])
    handle_types = ops.multinomial(dist, bs)
    zero_mask = handle_types == 0
    keep_mask = handle_types == 1

    return zero_mask, keep_mask
    

class DropPathWithControl(nn.Cell):
    """DropPath (Stochastic Depth) regularization with determinstic mask 
    Example:
       bs = 8
       batch_zero_control = ms.numpy.rand((bs, 1)) < p_all_zero
       batch_keep_control = ms.numpy.rand((bs, 1)) < p_all_keep
       dpc = DropPathWithControl(drop_prob=0.5)
       x = ms.ops.ones((bs, 4))
       dpc(x, zero_mask=batch_zero_control, keep_mask=batch_keep_control)
    """
    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        self.dropout = nn.Dropout(p=drop_prob)
        self.scale_by_keep = scale_by_keep
        self.cast = ops.Cast()

    def construct(self, x: ms.Tensor, zero_mask=None, keep_mask=None) -> ms.Tensor:
        '''
        x: (batch_size, ...)
        zero_mask: 1-d array in shape (batch_size, ), e.g. [1, 0, 0, 0], where index of 1 will be used to zero-out to the corresponding sample in a batch, and index of 0 will has no effect and leave the droppath randomness for the sample.
        keep_mask: 1-d array in shape (batch_size, ), where the index of 1 will be used to force the corresponding sample to be kept in output. 
            For "1" overlap in zero_mask and keep_mask, zero_mask has high priority.
        '''
        if self.keep_prob == 1.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ones(shape)) #* self.keep_prob

        if keep_mask is not None:
            random_tensor = random_tensor + keep_mask.view(shape)
        if zero_mask is not None:
            random_tensor = random_tensor * (1 - zero_mask.view(shape))
            
        random_tensor = self.cast((random_tensor > 0), x.dtype) / self.keep_prob

        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        
        return x * random_tensor


class DropPathFromVCPT(nn.Cell):
    r"""DropPath but without rescaling and supports optional all-zero and/or all-keep."""

    def __init__(self, p):
        super(DropPath, self).__init__()
        self.p = p

    def construct(self, *args, zero=None, keep=None):
        if not self.training:
            return args[0] if len(args) == 1 else args

        # params
        x = args[0]
        b = x.shape[0]
        n = (ops.rand(b) < self.p).sum()

        # non-zero and non-keep mask
        mask = x.new_ones(b, dtype=ms.bool_)
        if keep is not None:
            mask[keep] = False
        if zero is not None:
            mask[zero] = False

        # drop-path index
        index = ops.nonzero(mask).t()[0]  # special case for ops.nonzero, that the input is 1-d tensor
        index = index[ops.randperm(len(index))[:n]]
        if zero is not None:
            index = ops.cat([index, ops.nonzero(zero).t()[0]], axis=0)

        # drop-path multiplier
        multiplier = x.new_ones(b)
        multiplier[index] = 0.0
        output = tuple(u * self.broadcast(multiplier, u) for u in args)
        return output[0] if len(args) == 1 else output

    def broadcast(self, src, dst):
        assert src.shape[0] == dst.shape[0]
        shape = (dst.shape[0],) + (1,) * (dst.ndim - 1)
        return src.view(shape)

class DropPath(nn.Cell):
    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = nn.Dropout(1-drop_prob) if is_old_ms_version() else nn.Dropout(p=drop_prob)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ones(shape))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor
