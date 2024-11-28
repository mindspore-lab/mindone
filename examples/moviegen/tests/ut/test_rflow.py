import numpy as np
from mg.schedulers import RFlowLossWrapper

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class SimpleBF16Net(nn.Cell):
    def construct(self, x: Tensor, timestamp: Tensor, text_embedding: Tensor):
        return x.to(ms.bfloat16)

    @property
    def dtype(self):
        return ms.bfloat16


def test_rflow_loss():
    ms.set_context(mode=ms.GRAPH_MODE)
    network = RFlowLossWrapper(
        SimpleBF16Net(), num_timesteps=1000, sample_method="logit-normal", loc=0.0, scale=1.0, eps=1e-5
    )

    latent_embedding = ms.Tensor(np.ones((2, 16, 8, 24, 44)), dtype=ms.bfloat16)
    text_embedding = ms.Tensor(np.ones((2, 64, 4096)), dtype=ms.bfloat16)
    loss = network(latent_embedding, text_embedding).item()
    assert loss > 0
