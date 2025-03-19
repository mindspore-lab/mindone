import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops


class Normalization(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.mean = Parameter(Tensor(np.zeros(shape), ms.float32), requires_grad=False)
        self.variance = Parameter(Tensor(np.ones(shape), ms.float32), requires_grad=False)

    def construct(self, x):
        return (x - self.mean) / ops.sqrt(self.variance)


class NSFWModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.norm = Normalization(768)
        self.linear_1 = nn.Dense(768, 64)
        self.linear_2 = nn.Dense(64, 512)
        self.linear_3 = nn.Dense(512, 256)
        self.linear_4 = nn.Dense(256, 1)
        self.act = nn.ReLU()
        self.act_out = nn.Sigmoid()

    def construct(self, x):
        x = self.norm(x)
        x = self.act(self.linear_1(x))
        x = self.act(self.linear_2(x))
        x = self.act(self.linear_3(x))
        x = self.act_out(self.linear_4(x))
        return x
