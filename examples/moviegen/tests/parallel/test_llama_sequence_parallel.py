import argparse
from typing import Tuple

import numpy as np
from mg.acceleration import create_parallel_group, get_sequence_parallel_group
from mg.models.llama.network import LlamaModel

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication import get_group_size, init


class MeanNet(nn.Cell):
    def __init__(self, net: nn.Cell) -> None:
        super().__init__()
        self.net = net

    def construct(self, *inputs):
        output = self.net(*inputs)
        return output.mean() * 1024.0


def get_sample_data(dtype: ms.Type = ms.float32) -> Tuple[Tensor, ...]:
    latent_embedding = ops.rand([1, 16, 8, 24, 44], dtype=dtype)
    timestep = ms.Tensor([35], dtype=ms.int64)
    ul2_emb = ops.rand([1, 64, 4096], dtype=dtype)
    metaclip_emb = ops.rand([1, 64, 1280], dtype=dtype)
    byt5_emb = ops.rand([1, 64, 1472], dtype=dtype)
    return latent_embedding, timestep, ul2_emb, metaclip_emb, byt5_emb


def get_network_config():
    config = dict(num_hidden_layers=1, attn_implementation="eager", post_init_weight=False)
    return config


def run_network(mode: int = 0, dtype: ms.Type = ms.float32):
    ms.set_context(mode=mode)
    init()

    # prepare data
    ms.set_seed(1024)
    data = get_sample_data(dtype=dtype)

    run_parallel_network(data, dtype=dtype)


def run_parallel_network(data: Tuple[Tensor, ...], dtype: ms.Type = ms.float32):
    # non parallel network
    ms.set_seed(1024)
    non_parallel_network_cfg = get_network_config()
    non_parallel_network = LlamaModel(**non_parallel_network_cfg, dtype=dtype)

    # parallel netowrk
    ms.set_seed(1024)
    create_parallel_group(shards=get_group_size())
    parallel_network_cfg = get_network_config()
    parallel_network = LlamaModel(**parallel_network_cfg, dtype=dtype)

    # load weight
    for (_, w0), (_, w1) in zip(non_parallel_network.parameters_and_names(), parallel_network.parameters_and_names()):
        w1.set_data(w0)  # FIXME: seed does not work
        np.testing.assert_allclose(w0.value().asnumpy(), w1.value().asnumpy())

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
    grad_fn = ms.grad(non_parallel_mean_net, grad_position=None, weights=non_parallel_mean_net.trainable_params())
    non_parallel_grads = grad_fn(*data)

    grad_fn = ms.grad(parallel_mean_net, grad_position=None, weights=parallel_mean_net.trainable_params())
    parallel_grads = grad_fn(*data)

    # take mean around different ranks
    sp_group = get_sequence_parallel_group()
    reduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=sp_group)
    num = get_group_size()
    syn_parallel_grads = list()
    for x in parallel_grads:
        syn_parallel_grads.append(reduce(x) / num)

    pass_grads = []
    for grad_0, grad_1 in zip(non_parallel_grads, syn_parallel_grads):
        is_passed = np.allclose(grad_0.asnumpy(), grad_1.asnumpy(), rtol=1.3e-6, atol=1e-5)
        pass_grads.append(is_passed)
    assert all(pass_grads), f"Pass rate ({sum(pass_grads)/len(pass_grads) * 100:.3f} %) is not 100 %"

    print("Test 2 (Backward: Parameter Gradient): Passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default=0, type=int, choices=[0, 1], help="Mode to test. (0: Graph Mode; 1: Pynative mode)"
    )
    args = parser.parse_args()
    run_network(mode=args.mode)
