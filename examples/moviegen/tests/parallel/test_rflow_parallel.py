import argparse
from typing import Tuple

from mg.parallel import create_parallel_group
from mg.schedulers import RFlowLossWrapper

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.communication import get_group_size, init

from mindone.utils.seed import set_random_seed


class SimpleNet(nn.Cell):
    def construct(
        self, x: Tensor, timestamp: Tensor, ul2_emb: Tensor, metaclip_emb: Tensor, byt5_emb: Tensor
    ) -> Tensor:
        return x.to(ms.float32)

    @property
    def dtype(self):
        return ms.float32


def get_sample_data(dtype: ms.Type = ms.float32) -> Tuple[Tensor, ...]:
    latent_embedding = ops.rand([1, 16, 8, 24, 44], dtype=dtype)
    ul2_emb = ops.rand([1, 64, 4096], dtype=dtype)
    metaclip_emb = ops.rand([1, 64, 1280], dtype=dtype)
    byt5_emb = ops.rand([1, 64, 1472], dtype=dtype)
    return latent_embedding, ul2_emb, metaclip_emb, byt5_emb


def run_network(mode: int = 0):
    ms.set_context(mode=mode)
    init()

    # prepare data
    set_random_seed(1024)
    data = get_sample_data()

    # prepare group
    create_parallel_group(model_parallel_shards=get_group_size())

    model = SimpleNet()

    # parallel netowrk
    network = RFlowLossWrapper(model)

    loss = network(*data)
    loss = ops.AllGather()(ops.unsqueeze(loss, 0)).asnumpy()
    assert loss[0] == loss[1], f"expected two elements to be same, but get `{loss}`."
    print("Test 1: Passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default=0, type=int, choices=[0, 1], help="Mode to test. (0: Graph Mode; 1: Pynative mode)"
    )
    args = parser.parse_args()
    run_network(mode=args.mode)
