import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer, Normal


class FastLinear(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 has_bias=True,
                 weight_init="normal",
                 bias_init="zeros",
                 transpose_b=True,
                 param_init_type=mindspore.float32,
                 compute_dtype=mindspore.float16,
                 init_method_std=0.01
                 ):
        super(FastLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        if isinstance(weight_init, Tensor) and (weight_init.ndim != 2 or weight_init.shape[0] != weight_shape[0] or
                                                weight_init.shape[1] != weight_shape[1]):
            raise ValueError("The shape of parameter 'weight_init' is error, please check shape of 'weight_init'.")

        self.transpose_b = transpose_b
        if weight_init == "normal":
            weight_init = Normal(sigma=init_method_std, mean=0)
        self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")
        self.matmul = P.MatMul(transpose_b=transpose_b)
        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            if isinstance(bias_init, Tensor) and (bias_init.ndim != 1 or bias_init.shape[0] != out_channels):
                raise ValueError("The shape of parameter 'bias_init' is error, please check shape of 'bias_init'.")
            self.bias = Parameter(initializer(bias_init, [out_channels], param_init_type), name="bias")
            self.bias_add = P.Add()

        self.param_init_type = param_init_type
        self.dtype = compute_dtype
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, x):
        """Forward process, x should be a tensor"""
        out_shape = self.shape(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        x = self.cast(x, self.dtype)
        # apply gmm to the inference of moe structural models when use_past=True.
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        x = F.cast(x, ori_dtype)
        output = self.reshape(x, out_shape)
        return output
