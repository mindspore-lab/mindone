import argparse

import numpy as np
from llama.models.llama.block import LlamaMLP, TensorParallelLlamaMLP
from llama.parallel.parallel_states import create_parallel_group, get_model_parallel_group

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication import get_group_size, get_rank, init

from mindone.utils.seed import set_random_seed


class MeanNet(nn.Cell):
    def __init__(self, net: nn.Cell) -> None:
        super().__init__()
        self.net = net

    def construct(self, *inputs):
        output = self.net(*inputs)
        return output.mean()


def get_sample_data() -> Tensor:
    x = ops.rand([4, 64, 3072], dtype=ms.float32)  # (N, T, H)
    return x


def get_block_config():
    config = dict(intermediate_size=8192, hidden_size=3072, hidden_act="silu")
    return config


def run_block(mode: int = 0, dtype: ms.Type = ms.float32):
    ms.set_context(mode=mode)
    init()

    # prepare data
    set_random_seed(1024)
    data = get_sample_data()

    # prepare group
    create_parallel_group(model_parallel_shards=get_group_size())

    # non parallel block
    set_random_seed(1024)
    non_parallel_block_cfg = get_block_config()
    non_parallel_block = LlamaMLP(**non_parallel_block_cfg, dtype=dtype)

    # parallel block
    group = get_model_parallel_group()
    set_random_seed(1024)
    parallel_block_cfg = get_block_config()
    parallel_block = TensorParallelLlamaMLP(**parallel_block_cfg, dtype=dtype)

    tp_size = get_group_size(group)
    tp_rank = get_rank(group)
    for (_, w0), (_, w1) in zip(non_parallel_block.parameters_and_names(), parallel_block.parameters_and_names()):
        if w0.name.startswith("gate_proj") or w0.name.startswith("up_proj"):
            w0_col = ops.chunk(w0, tp_size, axis=0)[tp_rank]
            w1.set_data(w0_col)
        else:
            w0_row = ops.chunk(w0, tp_size, axis=1)[tp_rank]
            w1.set_data(w0_row)

    # test forward
    non_parallel_out = non_parallel_block(data)
    parallel_out = parallel_block(data)

    np.testing.assert_equal(non_parallel_out.shape, parallel_out.shape)
    np.testing.assert_allclose(non_parallel_out.asnumpy(), parallel_out.asnumpy(), atol=1e-5)
    print("Test 1 (Forward): Passed.")

    # test backward
    non_parallel_mean_net = MeanNet(non_parallel_block)
    parallel_mean_net = MeanNet(parallel_block)

    # check the input gradient
    grad_fn = ops.grad(non_parallel_mean_net, grad_position=0)
    non_parallel_grads = grad_fn(data)

    grad_fn = ops.grad(parallel_mean_net, grad_position=0)
    parallel_grads = grad_fn(data)

    for grad_0, grad_1 in zip(non_parallel_grads, parallel_grads):
        np.testing.assert_allclose(grad_0.asnumpy(), grad_1.asnumpy(), atol=1e-5)
    print("Test 2 (Backward: Input Gradient): Passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default=0, type=int, choices=[0, 1], help="Mode to test. (0: Graph Mode; 1: Pynative mode)"
    )
    args = parser.parse_args()
    run_block(mode=args.mode)
