import argparse
from typing import Tuple

import numpy as np
from moviegen.models.llama.network import LlamaModel
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


def get_sample_data(dtype: ms.Type = ms.float32) -> Tuple[Tensor, Tensor, Tensor]:
    latent_embedding = ops.rand([1, 16, 8, 24, 44], dtype=dtype)
    timestep = ms.Tensor([35], dtype=ms.int64)
    text_embedding = ops.rand([1, 64, 4096], dtype=dtype)
    return latent_embedding, timestep, text_embedding


def get_network_config(model_parallelism=False, fused_tensor_parallel=False):
    config = dict(
        num_hidden_layers=2,
        attn_implementation="eager",
        model_parallelism=model_parallelism,
        fused_tensor_parallel=fused_tensor_parallel,
        post_init_weight=False,
    )
    return config


def run_network(mode: int = 0, fused_tensor_parallel: bool = False, dtype: ms.Type = ms.float32):
    ms.set_context(mode=mode)
    init()

    # prepare data
    set_random_seed(1024)
    data = get_sample_data(dtype=dtype)

    # prepare group
    create_parallel_group(model_parallel_shards=get_group_size())

    # non parallel network
    set_random_seed(1024)
    non_parallel_network_cfg = get_network_config(model_parallelism=False, fused_tensor_parallel=fused_tensor_parallel)
    non_parallel_network = LlamaModel(**non_parallel_network_cfg, dtype=dtype)

    # parallel netowrk
    parallel_network_cfg = get_network_config(model_parallelism=True, fused_tensor_parallel=fused_tensor_parallel)
    parallel_network = LlamaModel(**parallel_network_cfg, dtype=dtype)

    # load weight
    parallel_network.load_weight_from_non_parallel_cell(non_parallel_network)

    # test forward
    non_parallel_out = non_parallel_network(*data).asnumpy()
    parallel_out = parallel_network(*data).asnumpy()

    assert np.count_nonzero(non_parallel_out) > 0
    np.testing.assert_equal(non_parallel_out.shape, parallel_out.shape)
    np.testing.assert_allclose(non_parallel_out, parallel_out, rtol=1.3e-6, atol=1e-5)
    print("Test 1 (Forward): Passed.", flush=True)

    # test backward
    non_parallel_mean_net = MeanNet(non_parallel_network)
    parallel_mean_net = MeanNet(parallel_network)

    # check the parameter gradient
    grad_fn = ops.grad(non_parallel_mean_net, grad_position=None, weights=non_parallel_mean_net.trainable_params())
    non_parallel_grads = grad_fn(*data)

    grad_fn = ops.grad(parallel_mean_net, grad_position=None, weights=parallel_mean_net.trainable_params())
    parallel_grads = grad_fn(*data)

    for grad_0, grad_1 in zip(non_parallel_grads, parallel_grads):
        grad_1 = gather_or_reduce_parallel_gradient(grad_1, grad_0.shape)
        grad_0, grad_1 = grad_0.asnumpy(), grad_1.asnumpy()
        assert np.count_nonzero(grad_0) > 0
        np.testing.assert_allclose(grad_0, grad_1, rtol=1.3e-6, atol=2e-5)
    print("Test 2 (Backward: Parameter Gradient): Passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default=0, type=int, choices=[0, 1], help="Mode to test. (0: Graph Mode; 1: Pynative mode)"
    )
    args = parser.parse_args()
    print("Non-fused tensor parallel:", flush=True)
    run_network(mode=args.mode, fused_tensor_parallel=False)

    print("Fused tensor parallel:", flush=True)
    run_network(mode=args.mode, fused_tensor_parallel=True)
