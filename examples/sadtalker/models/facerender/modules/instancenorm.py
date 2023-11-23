import mindspore as ms
import numpy as np
from mindspore import Parameter, Tensor, ops, nn
from mindspore.common import initializer


class InstanceNorm2d(nn.Cell):
    """mindediting InstanceNorm2d"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, gamma_init="ones", beta_init="zeros"):
        super().__init__()
        self.num_features = num_features
        self.moving_mean = Parameter(initializer.initializer(
            "zeros", num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(
            initializer.initializer("ones", num_features), name="variance", requires_grad=False
        )
        self.gamma = Parameter(initializer.initializer(
            gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = Parameter(initializer.initializer(
            beta_init, num_features), name="beta", requires_grad=affine)
        self.sqrt = ops.Sqrt()
        self.eps = Tensor(np.array([eps]), ms.float32)
        self.cast = ops.Cast()

    def construct(self, x):
        """calculate InstanceNorm output"""
        mean = ops.ReduceMean(keep_dims=True)(x, (2, 3))
        mean = self.cast(mean, ms.float32)
        tmp = x - mean
        tmp = tmp * tmp
        var = ops.ReduceMean(keep_dims=True)(tmp, (2, 3))
        std = self.sqrt(var + self.eps)
        gamma_t = self.cast(self.gamma, ms.float32)
        beta_t = self.cast(self.beta, ms.float32)
        x = (x - mean) / std * gamma_t.reshape(1, -
                                               1, 1, 1) + beta_t.reshape(1, -1, 1, 1)
        return x
