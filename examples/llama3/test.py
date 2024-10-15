import numpy as np
from llama.models import llama3_8B

import mindspore as ms
import mindspore.nn as nn


def count_params(model: nn.Cell) -> int:
    total_params = sum([param.size for param in model.get_parameters()])
    return total_params


def main():
    ms.set_context(mode=ms.GRAPH_MODE)
    network = llama3_8B(attn_implementation="flash_attention", dtype=ms.bfloat16)

    params = count_params(network)
    print(f"Parameter number: {params:,}")

    latent_embedding = ms.Tensor(np.ones((1, 16, 8, 24, 44)), dtype=ms.bfloat16)
    text_embedding = ms.Tensor(np.ones((1, 64, 256)), dtype=ms.bfloat16)
    outputs = network(latent_embedding, text_embedding)

    print(outputs.shape)
    print(outputs)


if __name__ == "__main__":
    main()
