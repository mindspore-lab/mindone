import os, sys
import mindspore as ms
import numpy as np
from mindspore import context

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from videogvt.models.quantization import LFQ
from videogvt.config.vqgan3d_ucf101_config import get_config

# GRAPH_MODE
# PYNATIVE_MODE

context.set_context(
    mode=context.GRAPH_MODE, device_target="Ascend", device_id=1
)

config = get_config("B")
input_dim = config.vqvae.middle_channles
n_embeddings = config.vqvae.codebook_size

model = LFQ(
    dim=input_dim,
    codebook_size=n_embeddings,
    return_loss_breakdown=False,
    is_training=True,
)

x = ms.Tensor(np.random.rand(2, 18, 4, 16, 16), ms.float32)

z_q, indices, aux_loss = model(x)

print("original input shape: ", x.shape)
print("quantized shape: ", z_q.shape)
print("indeces shape: ", indices.shape)
