import argparse
from typing import Union

import numpy as np
from einops import repeat
from opensora.acceleration import create_parallel_group, get_sequence_parallel_group
from opensora.models.mmdit.model import Flux

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, tensor
from mindspore.communication import get_group_size, init


class MeanNet(nn.Cell):
    def __init__(self, net: nn.Cell) -> None:
        super().__init__()
        self.net = net

    def construct(self, *inputs):
        output = self.net(*inputs)
        return output.mean() * 1024.0


def get_sample_data(t: int, h: int, w: int, dtype: ms.Type = ms.float32) -> tuple[Union[Tensor, None], ...]:
    video = tensor(np.random.randn(1, t * h * w, 64), dtype=dtype)
    video_ids = np.zeros((t, h, w, 3), dtype=np.int32)
    video_ids[..., 0] = video_ids[..., 0] + np.arange(t)[:, None, None]
    video_ids[..., 1] = video_ids[..., 1] + np.arange(h)[None, :, None]
    video_ids[..., 2] = video_ids[..., 2] + np.arange(w)[None, None, :]
    video_ids = tensor(repeat(video_ids, "t h w c -> 1 (t h w) c"), dtype=dtype)

    txt = tensor(np.random.randn(1, 512, 4096), dtype=dtype)
    txt_ids = tensor(np.zeros((1, 512, 3)), dtype=dtype)
    timestep = tensor([35], dtype=dtype)
    y_vec = tensor(np.random.randn(1, 768), dtype=dtype)
    guidance = tensor([4], dtype=dtype)
    return video, video_ids, txt, txt_ids, timestep, y_vec, None, guidance


def get_network_config():
    config = dict(
        guidance_embed=False,
        fused_qkv=False,
        use_liger_rope=True,
        in_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        cond_embed=False,
        recompute_every_nth_block=1,
        attn_type="eager",
    )
    return config


def run_network(mode: int = 0, dtype="fp32"):
    ms.set_context(mode=mode)
    ms.set_deterministic(True)
    init()

    # prepare data
    ms.set_seed(1024)
    data_type = {"fp32": ms.float32, "fp16": ms.float16, "bf16": ms.bfloat16}[dtype]
    data = get_sample_data(2, 12, 21, dtype=data_type)

    run_parallel_network(data, dtype=dtype)


def run_parallel_network(data: dict[str, Tensor], dtype: ms.Type = ms.float32):
    # non parallel network
    ms.set_seed(1024)
    non_parallel_network_cfg = get_network_config()
    non_parallel_network = Flux(**non_parallel_network_cfg, dtype=dtype)

    # parallel netowrk
    ms.set_seed(1024)
    create_parallel_group(shards=get_group_size())
    parallel_network_cfg = get_network_config()
    parallel_network = Flux(**parallel_network_cfg, dtype=dtype)

    # load weight
    for (_, w0), (_, w1) in zip(non_parallel_network.parameters_and_names(), parallel_network.parameters_and_names()):
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
