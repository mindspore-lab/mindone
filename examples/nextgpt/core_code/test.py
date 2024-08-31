# import torch
# out = torch.where(torch.tensor(835)==torch.tensor([[1,835]]))
# print(out)
# print(type(out))
# import mindspore
# out = mindspore.ops.where(mindspore.tensor([[False,False]]),mindspore.tensor(1),mindspore.tensor(0))
# print((out))
# print(type(out))
# print(mindspore.tensor([[False,True]]).any())
#
import mindspore
from mindspore import Tensor, ops
import numpy as np
input = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
output = ops.silu(input)
print(output)

