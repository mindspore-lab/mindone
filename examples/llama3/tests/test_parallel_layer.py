import argparse

import numpy as np
from llama.parallel import ColumnParallelLinear
from llama.parallel.parallel_states import create_parallel_group, get_model_parallel_group

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.communication import get_group_size, get_rank, init

from mindone.utils.seed import set_random_seed


class MeanNet(nn.Cell):
    def __init__(self, net: nn.Cell) -> None:
        super().__init__()
        self.net = net

    def construct(self, *inputs):
        output = self.net(*inputs)
        return output.mean()


def get_sample_data():
    x = ops.rand([4, 64, 256], dtype=ms.float32)  # (N, T, H)
    return x


def get_layer_config():
    config = dict(in_features=256, out_features=32, bias=True)
    return config


def run_layer(mode: int = 0, dtype: ms.Type = ms.float32):
    ms.set_context(mode=mode)
    ms.set_auto_parallel_context(enable_alltoall=True)
    init()

    # prepare data
    set_random_seed(1024)
    data = get_sample_data()

    # non parallel layer
    set_random_seed(1024)
    non_parallel_layer_cfg = get_layer_config()
    non_parallel_layer = mint.nn.Linear(**non_parallel_layer_cfg, dtype=dtype)

    # parallel layer
    create_parallel_group(get_group_size())
    group = get_model_parallel_group()
    set_random_seed(1024)
    parallel_layer_cfg = get_layer_config()
    parallel_layer = ColumnParallelLinear(**parallel_layer_cfg, gather_output=True, group=group, dtype=dtype)

    mp_size = get_group_size(group)
    mp_rank = get_rank(group)
    for (_, w0), (_, w1) in zip(non_parallel_layer.parameters_and_names(), parallel_layer.parameters_and_names()):
        w0_p = ops.chunk(w0, mp_size, axis=0)[mp_rank]
        w1.set_data(w0_p)

    # test forward
    non_parallel_out = non_parallel_layer(data)
    parallel_out = parallel_layer(data)

    np.testing.assert_equal(non_parallel_out.shape, parallel_out.shape)
    np.testing.assert_allclose(non_parallel_out.asnumpy(), parallel_out.asnumpy(), atol=1e-5)
    print("Test 1 (Forward): Passed.")

    # test backward
    non_parallel_mean_net = MeanNet(non_parallel_layer)
    parallel_mean_net = MeanNet(parallel_layer)

    # check the parameter gradient
    grad_fn = ops.grad(non_parallel_mean_net, grad_position=None, weights=non_parallel_mean_net.trainable_params())
    non_parallel_grads = grad_fn(data)

    grad_fn = ops.grad(parallel_mean_net, grad_position=None, weights=parallel_mean_net.trainable_params())
    parallel_grads = grad_fn(data)

    allgather = ops.AllGather(group=group)
    syn_parallel_grads = list()
    for x in parallel_grads:
        syn_parallel_grads.append(allgather(x))

    for grad_0, grad_1 in zip(non_parallel_grads, syn_parallel_grads):
        np.testing.assert_allclose(grad_0.asnumpy(), grad_1.asnumpy(), atol=1e-5)
    print("Test 2 (Backward: Parameter Gradient): Passed.")

    # check the input gradient
    grad_fn = ops.grad(non_parallel_mean_net, grad_position=0)
    non_parallel_grads = grad_fn(data)

    grad_fn = ops.grad(parallel_mean_net, grad_position=0)
    parallel_grads = grad_fn(data)

    for grad_0, grad_1 in zip(non_parallel_grads, parallel_grads):
        np.testing.assert_allclose(grad_0.asnumpy(), grad_1.asnumpy(), atol=1e-5)
    print("Test 3 (Backward: Input Gradient): Passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default=0, type=int, choices=[0, 1], help="Mode to test. (0: Graph Mode; 1: Pynative mode)"
    )
    args = parser.parse_args()
    run_layer(mode=args.mode)
