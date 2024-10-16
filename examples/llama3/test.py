import numpy as np
from llama.models import llama3_1B

import mindspore as ms
import mindspore.nn as nn


def count_params(model: nn.Cell) -> int:
    total_params = sum([param.size for param in model.get_parameters()])
    return total_params


def main():
    ms.set_context(mode=ms.GRAPH_MODE)
    network = llama3_1B(attn_implementation="flash_attention", dtype=ms.bfloat16)

    params = count_params(network)
    print(f"Parameter number: {params:,}")

    latent_embedding = ms.Tensor(np.ones((1, 16, 8, 24, 44)), dtype=ms.bfloat16)
    timestep = ms.Tensor([35], dtype=ms.int64)
    text_embedding = ms.Tensor(np.ones((1, 64, 4096)), dtype=ms.bfloat16)
    outputs = network(latent_embedding, timestep, text_embedding)

    print(outputs.shape)


if __name__ == "__main__":
    main()
