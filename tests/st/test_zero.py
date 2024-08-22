import logging

import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore.communication import get_group_size, get_rank, init
from mindspore.communication.management import GlobalComm

from mindone.trainers.zero import prepare_train_network
from mindone.utils.logger import set_logger

_logger = logging.getLogger(__name__)


def init_env(mode, distribute, save_graph=True, comm_fusio=False):
    ms.set_seed(1)
    ms.set_context(mode=mode)
    if save_graph:
        ms.set_context(save_graphs=True, save_graphs_path="ms_ir")
    if distribute:
        init()
        group_size = get_group_size()
        rank_id = get_rank()
        print(f"rank_id: {rank_id}, group_size: {group_size}")
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
        )
        if comm_fusio:
            comm_fusion_dict = {"allreduce": {"mode": "auto", "config": None},
                                "reducescatter": {"mode": "auto", "config": None},
                                "allgather": {"mode": "auto", "config": None},}
            ms.set_auto_parallel_context(comm_fusion=comm_fusion_dict)
        return group_size, rank_id
    return 1, 0


class TestNet(nn.Cell):
    def __init__(self, p=1):
        super().__init__()
        self.p = p
        self.conv1 = nn.Conv2d(2, 4, 1, has_bias=True)
        self.dense = nn.Dense(4, 4)
        self.conv2 = nn.Conv2d(4, 2, 1, has_bias=True)
        self.conv1.bias.parallel_optimizer = False
        self.conv2.bias.parallel_optimizer = False
        self.dense.recompute()
        self.conv1.to_float(ms.float16)
        self.dense.to_float(ms.float32)

    def construct(self, x):
        y = self.conv1(x) * self.p
        y = y.transpose((0, 2, 3, 1))
        y = self.dense(y)
        y = y.transpose((0, 3, 1, 2))
        y = self.conv2(y)
        return y


def test_zero(x, y, zero_stage=0, comm_fusion=False):
    print("-" * 30)
    print("-" * 6, f"zero_stage={zero_stage}", "-" * 6)
    print("-" * 30)
    ms.set_seed(1)
    net = nn.WithLossCell(TestNet(), nn.MSELoss())
    opt = nn.AdamWeightDecay(net.trainable_params(), learning_rate=1e-3)
    comm_fusion_dict = None
    if comm_fusion:
        comm_fusion_dict = {"allreduce": {"openstate": True, "bucket_size": 5e8},
                            "reduce_scatter": {"openstate": True, "bucket_size": 5e8},
                            "allgather": {"openstate": False, "bucket_size": 5e8},}
    train_net = prepare_train_network(net, opt, zero_stage=zero_stage, op_group=GlobalComm.WORLD_COMM_GROUP,
                                      comm_fusion=comm_fusion_dict)

    for i in range(10):
        loss = train_net(x, y)
        print(f"== {i} == loss", loss)


if __name__ == "__main__":
    group_size, rank_id = init_env(mode=0, distribute=True, save_graph=True)
    set_logger(name="", output_dir="logs", rank=rank_id, log_level="DEBUG")
    x = ms.Tensor(np.random.uniform(-1, 1, (1, 2, 5, 5)).astype(np.float32) * (get_rank() + 1))
    y = ms.Tensor(np.random.uniform(-1, 1, (1, 2, 5, 5)).astype(np.float32) * (get_rank() + 1))
    test_zero(x, y, zero_stage=0)
    test_zero(x, y, zero_stage=1)
    test_zero(x, y, zero_stage=2)
    test_zero(x, y, zero_stage=3)
