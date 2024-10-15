import numpy as np
from llama.models import LlamaModel

import mindspore as ms
import mindspore.nn as nn


def count_params(model: nn.Cell) -> int:
    total_params = sum([param.size for param in model.get_parameters()])
    return total_params


def main():
    ms.set_context(mode=ms.GRAPH_MODE)
    network = LlamaModel(attn_implementation="flash_attention", dtype=ms.bfloat16)
    ms.load_checkpoint("model.ckpt", network)

    params = count_params(network)
    print(f"Parameter number: {params:,}")

    inputs = ms.Tensor(np.ones((4, 256, 4096)), dtype=ms.bfloat16)
    outputs = network(inputs)

    print(outputs.shape)
    print(outputs)


if __name__ == "__main__":
    main()
