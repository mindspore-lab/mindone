import os, sys
import mindspore as ms
import numpy as np
from mindspore import context

# GRAPH_MODE - 0
# PYNATIVE_MODE - 1
context.set_context(mode=0, device_target="Ascend", device_id=1)

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from videogvt.models.transformer import MAGVITransformer, Transformer

model = Transformer(24, 512, 30, depth=2, add_mask_id=True)
tokens = ms.Tensor(np.random.randint(100, size=(2, 24)), ms.int32)
texts = ["test codes for the transformer 1!", "test codes for the transformer 2!"]
logits = model(tokens, texts=texts)

print(logits.shape)

model = MAGVITransformer(num_tokens=24, dim=512, seq_len=30, depth=2)
logits = model(tokens, texts=texts)

print(logits.shape)
