import argparse

import numpy as np
from moviegen.models.llama.block import LlamaMLP, TensorParallelLlamaMLP
from moviegen.parallel import create_parallel_group
from utils import gather_or_reduce_parallel_gradient

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication import get_group_size, init

from mindone.utils.seed import set_random_seed


class MeanNet(nn.Cell):
    def __init__(self, net: nn.Cell) -> None:
        super().__init__()
        self.net = net

    def construct(self, *inputs):
        output = self.net(*inputs)
        return output.mean() * 1024.0


def get_sample_data(dtype: ms.Type = ms.float32) -> Tensor:
    x = ops.rand([4, 64, 3072], dtype=dtype)  # (N, T, H)
    return x


def get_block_config():
    config = dict(intermediate_size=8192, hidden_size=3072, hidden_act="silu")
    return config


def run_block(mode: int = 0, dtype: ms.Type = ms.float32):
    ms.set_context(mode=mode)
    init()

    # prepare data
    set_random_seed(1024)
    data = get_sample_data(dtype=dtype)

    # prepare group
    create_parallel_group(model_parallel_shards=get_group_size())

    # non parallel block
    set_random_seed(1024)
    non_parallel_block_cfg = get_block_config()
    non_parallel_block = LlamaMLP(**non_parallel_block_cfg, dtype=dtype)

    # parallel block
    parallel_block_cfg = get_block_config()
    parallel_block = TensorParallelLlamaMLP(**parallel_block_cfg, dtype=dtype)

    # load weight
    parallel_block.load_weight_from_non_parallel_cell(non_parallel_block)

    # test forward
    non_parallel_out = non_parallel_block(data).asnumpy()
    parallel_out = parallel_block(data).asnumpy()

    assert np.count_nonzero(non_parallel_out) > 0
    np.testing.assert_equal(non_parallel_out.shape, parallel_out.shape)
    np.testing.assert_allclose(non_parallel_out, parallel_out, rtol=1.3e-6, atol=1e-5)
    print("Test 1 (Forward): Passed.")

    # test backward
    non_parallel_mean_net = MeanNet(non_parallel_block)
    parallel_mean_net = MeanNet(parallel_block)

    # check the parameter gradient
    grad_fn = ops.grad(non_parallel_mean_net, grad_position=None, weights=non_parallel_mean_net.trainable_params())
    non_parallel_grads = grad_fn(data)

    grad_fn = ops.grad(parallel_mean_net, grad_position=None, weights=parallel_mean_net.trainable_params())
    parallel_grads = grad_fn(data)

    for grad_0, grad_1 in zip(non_parallel_grads, parallel_grads):
        grad_1 = gather_or_reduce_parallel_gradient(grad_1, grad_0.shape)
        grad_0, grad_1 = grad_0.asnumpy(), grad_1.asnumpy()
        assert np.count_nonzero(grad_0) > 0
        np.testing.assert_allclose(grad_0, grad_1, rtol=1.3e-6, atol=1e-5)
    print("Test 2 (Backward: Parameter Gradient): Passed.")

    # check the input gradient
    grad_fn = ops.grad(non_parallel_mean_net, grad_position=0)
    non_parallel_grads = grad_fn(data)

    grad_fn = ops.grad(parallel_mean_net, grad_position=0)
    parallel_grads = grad_fn(data)

    for grad_0, grad_1 in zip(non_parallel_grads, parallel_grads):
        grad_0, grad_1 = grad_0.asnumpy(), grad_1.asnumpy()
        assert np.count_nonzero(grad_0) > 0
        np.testing.assert_allclose(grad_0, grad_1, rtol=1.3e-6, atol=1e-5)
    print("Test 3 (Backward: Input Gradient): Passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default=0, type=int, choices=[0, 1], help="Mode to test. (0: Graph Mode; 1: Pynative mode)"
    )
    args = parser.parse_args()
    run_block(mode=args.mode)
