# Zero redundancy optimizer(ZeRO) on MindOne

Zero Redundancy Optimizer (ZeRO) is a method for reducing memory usage under data parallelism strategy on paper: [ZeRO: ZeRO: Memory Optimization Towards Training A Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf).

ZeRO eliminates memory redundancies in data and model parallel training while retaining low communication volume and high computational
granularity, allowing us to scale the model size proportional to the number of devices with sustained high efficiency.

This tutorial walks you through how to generate faster and better with the ZeRO on MindOne.

## Build Train Network With ZeRO

Build a train network with ZeRO.

```python
import mindspore as ms
from mindspore.communication import init
from mindspore.communication.management import GlobalComm
from mindone.trainers.zero import prepare_train_network

# Initialize distributed environment
def init_env(mode, distribute):
    ms.set_context(mode=mode)
    if distribute:
        init()
        # ZeRO take effect must on DATA_PARALLEL
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
        )

init_env(ms.GRAPH_MODE, True)

# Net is your Train Network
net = Net()
# opt must be the subclass of MindSpore Optimizer.
opt = nn.AdamWeightDecay(net.trainable_params(), learning_rate=1e-3)

# build a train network with ZeRO
train_net = prepare_train_network(net, opt, zero_stage=2, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
```

!!! tip
    optimizer_parallel_group may not be GlobalComm.WORLD_COMM_GROUP. Using [create_group](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.communication.html#mindspore.communication.create_group) to create your optimizer_parallel_group.

More details:

::: mindone.trainers.zero.prepare_train_network

[Here](https://github.com/mindspore-lab/mindone/blob/master/tests/others/test_zero.py) is an example.

## Memory Analysis

The memory consumption during the training can be divided into two main parts:

- Residual states. Mainly includes activate functions, temporary buffers, and unavailable memory fragments.
- Model states. Mainly includes three parts: optimizer states(AdamW fp32), gradients(fp16), and parameters(fp16). The three are abbreviated as OPG. Assuming the number of model parameters is Φ,
the total model states is 2Φ(parameters) + 2Φ(gradients) + (4Φ + 4Φ + 4Φ)(optimizer states) = 16Φ, the AdamW states accounting for 75%.

Residual states can be greatly reduced through [recompute](https://www.mindspore.cn/docs/en/master/model_train/parallel/recompute.html) and [model parallel](https://www.mindspore.cn/docs/en/master/model_train/parallel/strategy_select.html).
Then the ZeRO algorithm can be used to reduce model states.

For the optimization of model states (removing redundancy), ZeRO uses the method of partitioning, which means that each card only stores 1/N data.

ZeRO has three main optimization stages (as depicted in ZeRO paper Figure 1), which correspond to the partitioning of optimizer states, gradients, and parameters. When enabled cumulatively:

1) Optimizer State Partitioning (Pos): Optimizer states are kept 1/N, the model parameters and gradients are still kept in full on each card. The model state of each card is 4Φ + 12Φ/N, when N is very large, it tend to 4Φ, that's the 1/4 original memory;
2) Add Gradient Partitioning (Pos+g): Add the gradients partitioning to 1/N, The model state of each card is 2Φ + (2Φ + 12Φ)/N, when N is very large, it tend to 2Φ, that's the 1/8 original memory;
3) Add Parameter Partitioning (Pos+g+p): Add the parameters partitioning to 1/N, The model state of each card is 16Φ/N, when N is very large, it tend to 0;

Pos correspond to ZeRO-1, Pos+g correspond to ZeRO-2 and Pos+g+p correspond to ZeRO-3.

## Communitition Analysis

Currently, AllReduce commonly used method is Ring AllReduce, which is divided into two steps: ReduceScatter and AllGather. The communication data volume (send+receive) of each card is approximately 2Φ.

| zero stage | forward + backward | gradient            | optimizer update | communitition |
| --- |--------------------|---------------------|------------------|---------------|
| 0 | NA                 | AllReduce           | NA               | 2Φ            |
| 1 | NA                 | 1/N ReduceScatter       | 1/N AllGather  | 2Φ            |
| 2 | NA                 | 1/N ReduceScatter | 1/N AllGather  | 2Φ            |
| 3 | 2 AllGather        | ReduceScatter       | NA               | 3Φ            |

It can be concluded that Zero3 has an additional communication calculation. But, computing and communication are parallel streams on MindSpore. When the computation after communication is relatively large, ZeRO3 may be faster.

## CheckPoint Saving & Loading

Because the parameters of the model have been split, the parameters of each card need to be saved.

### Resume

checkpoint save:

| zero stage | parameters | optimizer states | ema |
|------------|------------| --- | --- |
| 0          | one card   |  one card |  one card |
| 1          | one card   |  each card |  each card |
| 2          | one card   |  each card |  each card |
| 3          | each card  |  each card |  each card |

!!! tip

    💡 Recommend using rank_id to distinguish checkpoint saved on different cards.

```python
rank_id = get_rank_id()
zero_stage=2
train_net = prepare_train_network(net, opt, zero_stage=zero_stage, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
if resume:
    network_ckpt = "network.ckpt" if zero_stage != 3 else f"network_{rank_id}.ckpt"
    ms.load_checkpoint(network_ckpt, net=train_net.network)
    optimizer_ckpt = "optimizer.ckpt" if zero_stage == 0 else f"optimizer_{rank_id}.ckpt"
    ms.load_checkpoint(optimizer_ckpt, net=train_net.optimizer)
    ema_ckpt = "ema.ckpt" if zero_stage == 0 else f"ema_{rank_id}.ckpt"
    ms.load_checkpoint(ema_ckpt, net=train_net.ema)
```

### Inference

Inference need complete model parameters when use zero3. There are two ways(online & offline) to get the complete model parameters.

#### Online Checkpoint Combile

```python
def do_ckpt_combine_online(net_to_save, optimizer_parallel_group):
    new_net_to_save = []
    all_gather_op = ops.AllGather(optimizer_parallel_group)
    for param in net_to_save:
        if param.parallel_optimizer:
            new_data = ms.Tensor(all_gather_op(param).asnumpy())
        else:
            new_data = ms.Tensor(param.asnumpy())
        new_net_to_save.append({"name": param.name, "data": new_data})
    return new_net_to_save

net_to_save = [{"name": p.name, "data": p} for p in network.trainable_params()]
net_to_save = net_to_save if zero_stage != 3 else do_ckpt_combine_online(net_to_save, optimizer_parallel_group)
ms.save_checkpoint(net_to_save, "network.ckpt")
```

Add the code when need save model parameters.

#### Offline Checkpoint Combile

Parameters split infomation will be save when using ZereHelper, could use it to combile the checkpoints offline.

```python
from mindone.trainers.zero import convert_checkpoints

src_checkpoint = "save_checkpoint_dir/ckpt_{}.ckpt"
src_param_split_info_json = "params_info/params_split_info_{}.json"
group_size = 2
convert_checkpoints(src_checkpoint, src_param_split_info_json, group_size)
```

And get the complete model parameters checkpoint at `save_checkpoint_dir/ckpt_all_2.ckpt`.
