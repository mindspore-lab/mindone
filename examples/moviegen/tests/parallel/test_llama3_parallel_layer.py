import argparse
from typing import Literal

import numpy as np
from moviegen.parallel import ColumnParallelLinear, RowParallelLinear, create_parallel_group, get_model_parallel_group
from utils import gather_or_reduce_parallel_gradient

import mindspore as ms
import mindspore.mint as mint
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
    x = ops.rand([4, 64, 256], dtype=dtype)  # (N, T, H)
    return x


def get_layer_config(bias: bool = False):
    config = dict(in_features=256, out_features=32, bias=bias)
    return config


def run_layer(mode: int = 0, dtype: ms.Type = ms.float32):
    ms.set_context(mode=mode)
    init()

    # prepare data
    set_random_seed(1024)
    data = get_sample_data(dtype=dtype)

    # prepare group
    create_parallel_group(model_parallel_shards=get_group_size())

    print("Column Parallel Linear (Bias=True):")
    run_parallel_linear(data, type="column_parallel", bias=True, dtype=dtype)
    print("Column Parallel Linear (Bias=False):")
    run_parallel_linear(data, type="column_parallel", bias=False, dtype=dtype)
    print("Row Parallel Linear (Bias=True):")
    run_parallel_linear(data, type="row_parallel", bias=True, dtype=dtype)
    print("Row Parallel Linear (Bias=False):")
    run_parallel_linear(data, type="row_parallel", bias=False, dtype=dtype)


def run_parallel_linear(
    data: Tensor, type: Literal["column_parallel", "row_parallel"], bias: bool = False, dtype: ms.Type = ms.float32
):
    # non parallel layer
    set_random_seed(1024)
    non_parallel_layer_cfg = get_layer_config(bias=bias)
    non_parallel_layer = mint.nn.Linear(**non_parallel_layer_cfg, dtype=dtype)

    # parallel layer
    group = get_model_parallel_group()
    parallel_layer_cfg = get_layer_config(bias=bias)
    if type == "column_parallel":
        parallel_layer = ColumnParallelLinear(**parallel_layer_cfg, gather_output=True, group=group, dtype=dtype)
    else:
        parallel_layer = RowParallelLinear(**parallel_layer_cfg, input_is_parallel=False, group=group, dtype=dtype)

    # load weight
    parallel_layer.load_weight_from_non_parallel_cell(non_parallel_layer)

    # test forward
    non_parallel_out = non_parallel_layer(data).asnumpy()
    parallel_out = parallel_layer(data).asnumpy()

    assert np.count_nonzero(non_parallel_out) > 0
    np.testing.assert_equal(non_parallel_out.shape, parallel_out.shape)
    np.testing.assert_allclose(non_parallel_out, parallel_out, rtol=1.3e-6, atol=1e-5)
    print("Test 1 (Forward): Passed.")

    # test backward
    non_parallel_mean_net = MeanNet(non_parallel_layer)
    parallel_mean_net = MeanNet(parallel_layer)

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
    run_layer(mode=args.mode)
