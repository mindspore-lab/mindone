import numpy as np
from mg import llama3_1B

import mindspore as ms


def test_llama3_forward_graph():
    ms.set_context(mode=ms.GRAPH_MODE)
    network = llama3_1B(attn_implementation="flash_attention", dtype=ms.bfloat16)

    latent_embedding = ms.Tensor(np.ones((1, 16, 8, 24, 44)), dtype=ms.bfloat16)
    timestep = ms.Tensor([35], dtype=ms.int64)
    text_embedding = ms.Tensor(np.ones((1, 64, 4096)), dtype=ms.bfloat16)
    outputs = network(latent_embedding, timestep, text_embedding)

    assert outputs.shape == (1, 16, 8, 24, 44)
