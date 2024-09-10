import os, sys
import mindspore as ms
import numpy as np
from mindspore import context

# GRAPH_MODE 0
# PYNATIVE_MODE 1

context.set_context(mode=0, device_target="Ascend", device_id=1)

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from videogvt.models.magvit import MAGVIT
from videogvt.models.vqvae.vqvae import VQVAE3D
from videogvt.models.transformer import MAGVITransformer
from videogvt.config.vqgan3d_ucf101_config import get_config


config = get_config("B")
vae = VQVAE3D(config, lookup_free_quantization=True, is_training=False)
transformer = MAGVITransformer(num_tokens=1024, dim=512, seq_len=1025, depth=2)

vae.set_train(False)
transformer.set_train(False)
model = MAGVIT(video_size=128, transformer=transformer, vae=vae)
fmap_size = 8
texts = ["test codes for the transformer 1!", "test codes for the transformer 2!"]
recon = model.generate(texts=texts, fmap_size=fmap_size)

print(recon)
