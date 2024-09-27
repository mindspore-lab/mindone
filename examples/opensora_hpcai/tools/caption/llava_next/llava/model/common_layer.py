"""Holding the layer where the parameter name and type is consistent with Pytorch"""
import numbers
from typing import List, Optional, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import Initializer, initializer

__all__ = ["LayerNorm", "Embedding"]


class LayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: Union[Tuple[int], List[int]],
        begin_norm_axis: int = -1,
        begin_params_axis: int = -1,
        gamma_init: Union[Tensor, str, Initializer, numbers.Number] = "ones",
        beta_init: Union[Tensor, str, Initializer, numbers.Number] = "zeros",
        epsilon: float = 1e-7,
        dtype: ms.dtype = ms.float32,
    ):
        """Initialize LayerNorm."""
        super(nn.LayerNorm, self).__init__()
        if not isinstance(normalized_shape, (tuple, list)):
            raise TypeError(
                f"For '{self.cls_name}', the type of 'normalized_shape' must be tuple[int] or list[int], "
                f"but got {normalized_shape} and the type is {type(normalized_shape)}."
            )
        if not normalized_shape:
            raise ValueError(
                f"Expected normalized_shape to be at least 1-dimensional, i.e., containing at "
                f"least one element, but got normalized_shape = {normalized_shape}"
            )
        self.normalized_shape = normalized_shape
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.epsilon = epsilon
        self.weight = Parameter(initializer(gamma_init, normalized_shape, dtype=dtype))
        self.bias = Parameter(initializer(beta_init, normalized_shape, dtype=dtype))
        self.layer_norm = ops.LayerNorm(
            begin_norm_axis=self.begin_norm_axis, begin_params_axis=self.begin_params_axis, epsilon=self.epsilon
        )

    def construct(self, input_x):
        y, _, _ = self.layer_norm(input_x, self.weight.astype(input_x.dtype), self.bias.astype(input_x.dtype))
        return y

    def extend_repr(self):
        return "normalized_shape={}, begin_norm_axis={}, begin_params_axis={}, gamma{}, beta={}".format(
            self.normalized_shape, self.begin_norm_axis, self.begin_params_axis, self.weight, self.bias
        )


class Embedding(nn.Embedding):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        use_one_hot: bool = False,
        embedding_table: Union[Tensor, str, Initializer, numbers.Number] = "normal",
        dtype: ms.dtype = ms.float32,
        padding_idx: Optional[int] = None,
    ):
        """Initialize Embedding."""
        super(nn.Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot = use_one_hot
        self.dtype = dtype
        self.init_tensor = initializer(embedding_table, [vocab_size, embedding_size], dtype=dtype)
        self.padding_idx = padding_idx
        if padding_idx is not None:
            self.padding_idx = padding_idx
            if isinstance(self.init_tensor, Tensor) and self.init_tensor.init is not None:
                self.init_tensor = self.init_tensor.init_data()
            self.init_tensor = self.init_tensor.asnumpy()
            self.init_tensor[self.padding_idx] = 0
            self.init_tensor = Tensor(self.init_tensor)
        self.weight = Parameter(self.init_tensor)
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
        self.concat = ops.Concat()

    def construct(self, ids):
        out_shape = self.get_shp(ids) + (self.embedding_size,)
        flat_ids = self.reshape_flat(ids, self.shp_flat)

        if self.use_one_hot:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.weight)
        else:
            output_for_reshape = self.gather(self.weight, flat_ids, 0)

        output = self.reshape(output_for_reshape, out_shape)
        return output

    def extend_repr(self):
        return (
            f"vocab_size={self.vocab_size}, embedding_size={self.embedding_size}, use_one_hot={self.use_one_hot}, "
            f"embedding_table={self.weight}, dtype={self.dtype}, padding_idx={self.padding_idx}"
        )
