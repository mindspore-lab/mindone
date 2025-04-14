import numpy as np

import mindspore as ms
import mindspore.common.initializer as init
from mindspore import Tensor, Parameter, nn, ops


class Embedding(nn.Cell):
    """
    Rename embedding_table to weight
    """

    def __init__(self, vocab_size, embedding_size, use_one_hot=False, weight='normal',
                 dtype=ms.float32, padding_idx=None):
        """Initialize Embedding."""
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot = use_one_hot
        self.dtype = dtype
        self.init_tensor = init.initializer(weight, [vocab_size, embedding_size])
        self.padding_idx = padding_idx
        if padding_idx is not None:
            self.padding_idx = padding_idx
            if isinstance(self.init_tensor, Tensor) and self.init_tensor.init is not None:
                self.init_tensor = self.init_tensor.init_data()
            self.init_tensor = self.init_tensor.asnumpy()
            self.init_tensor[self.padding_idx] = 0
            self.init_tensor = Tensor(self.init_tensor)
        self.weight = Parameter(self.init_tensor, name='weight')
        self.expand = ops.ExpandDims()
        self.reshape_flat = ops.Reshape()
        self.shp_flat = (-1,)
        self.gather = ops.Gather()
        self.one_hot = ops.OneHot()
        self.on_value = Tensor(1.0, self.dtype)
        self.off_value = Tensor(0.0, self.dtype)
        self.array_mul = ops.MatMul()
        self.reshape = ops.Reshape()
        self.get_shp = ops.Shape()
        self.get_tensor_shp = ops.TensorShape()
        self.concat = ops.Concat()

    def construct(self, ids):
        out_shape = self.get_shp(ids) + (self.embedding_size,)
        if ops.is_sequence_value_unknown(self.get_shp(ids)):
            out_shape = self.concat((self.get_tensor_shp(ids), Tensor([self.embedding_size])))
        flat_ids = self.reshape_flat(ids, self.shp_flat)

        if self.use_one_hot:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.weight)
        else:
            output_for_reshape = self.gather(self.weight, flat_ids, 0)

        output = self.reshape(output_for_reshape, out_shape)
        return output

    def extend_repr(self):
        s = 'vocab_size={}, embedding_size={}, use_one_hot={}, weight={}, dtype={}, padding_idx={}'.format(
            self.vocab_size, self.embedding_size, self.use_one_hot, self.weight, self.dtype, self.padding_idx)
        return s


def finfo(dtype):
    return np.finfo(ms.dtype_to_nptype(dtype))
