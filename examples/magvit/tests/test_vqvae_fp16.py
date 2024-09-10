import os, sys
import mindspore as ms
import numpy as np
from mindspore import context

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from videogvt.models.vqvae import VQVAE3D, StyleGANDiscriminator
from videogvt.config.vqgan3d_ucf101_config import get_config

# GRAPH_MODE - 0
# PYNATIVE_MODE - 1

context.set_context(
    mode=0, device_target="Ascend", device_id=1
)

config = get_config("B")
model = VQVAE3D(config, lookup_free_quantization=True, is_training=True, dtype=ms.float16)
discriminator = StyleGANDiscriminator(config, 128, 128, 16, dtype=ms.float16)

x = ms.Tensor(np.random.rand(2, 3, 16, 128, 128), ms.float16)

z_e, z_q, x_hat, aux_loss = model(x)

# embedding_loss, x_hat, z_e, z_q = model(x)
logit_true = discriminator(x)
logit_fake = discriminator(x_hat)

print("original input shape: ", x.shape)
print("encoded shape: ", z_e.shape)
print("quantized shape: ", z_q.shape)
print("reconstructed shape:", x_hat.shape)
print("quantization loss: ", aux_loss)
# print("x_discriminate shape: ", x_dis.shape)
print("logits of true sample: ", logit_true)
print("logits of fake sample: ", logit_fake)
