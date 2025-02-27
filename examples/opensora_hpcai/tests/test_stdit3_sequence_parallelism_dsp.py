import argparse

import numpy as np
from opensora.acceleration.parallel_states import create_parallel_group, get_sequence_parallel_group
from opensora.models.stdit.stdit3 import STDiT3
from opensora.models.stdit.stdit3_dsp import STDiT3_DSP
from opensora.utils.amp import auto_mixed_precision
from opensora.utils.model_utils import WHITELIST_OPS

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
        return output.mean()


def get_sample_data():
    x = ops.rand([1, 4, 15, 20, 28], dtype=ms.float32)  # (B, C, T, H, W)
    timestep = Tensor([924.0], dtype=ms.float32)
    y = ops.rand(1, 1, 300, 4096, dtype=ms.float32)
    mask = ops.ones([1, 300], dtype=ms.uint8)
    frames_mask = ops.ones([1, 15], dtype=ms.bool_)
    fps = Tensor([25.0], dtype=ms.float32)
    height = Tensor([166.0], ms.float32)
    width = Tensor([221.0], ms.float32)
    return dict(x=x, timestep=timestep, y=y, mask=mask, frames_mask=frames_mask, fps=fps, height=height, width=width)


def get_stdit3_config(enable_sequence_parallelism=False):
    config = {
        "caption_channels": 4096,
        "class_dropout_prob": 0.0,
        "depth": 1,
        "drop_path": 0.0,
        "enable_flashattn": False,
        "enable_layernorm_kernel": False,
        "enable_sequence_parallelism": enable_sequence_parallelism,
        "freeze_y_embedder": True,
        "hidden_size": 1152,
        "in_channels": 4,
        "input_size": [None, None, None],
        "input_sq_size": 512,
        "mlp_ratio": 4.0,
        "model_max_length": 300,
        "num_heads": 16,
        "only_train_temporal": False,
        "patch_size": [1, 2, 2],
        "pred_sigma": True,
        "qk_norm": True,
        "skip_y_embedder": False,
        "patchify_conv3d_replace": "conv2d",
    }
    return config


def run_model(mode: int = 0, model_dtype: ms.dtype = ms.float32):
    ms.set_context(mode=mode)
    ms.set_auto_parallel_context(enable_alltoall=True)
    init()

    # prepare data
    set_random_seed(1024)
    data = get_sample_data()

    # single model
    set_random_seed(1024)
    non_dist_model_cfg = get_stdit3_config(enable_sequence_parallelism=False)
    non_dist_model = STDiT3(**non_dist_model_cfg)
    if model_dtype != ms.float32 or mode == 0:
        non_dist_model = auto_mixed_precision(
            non_dist_model,
            amp_level="O2",
            dtype=model_dtype,
            custom_fp32_cells=WHITELIST_OPS,
        )

    # sequence parallel model
    create_parallel_group(get_group_size())
    set_random_seed(1024)
    dist_model_cfg = get_stdit3_config(enable_sequence_parallelism=True)
    dist_model = STDiT3_DSP(**dist_model_cfg)
    if model_dtype != ms.float32 or mode == 0:
        dist_model = auto_mixed_precision(
            dist_model,
            amp_level="O2",
            dtype=model_dtype,
            custom_fp32_cells=WHITELIST_OPS,
        )

    for (_, w0), (_, w1) in zip(non_dist_model.parameters_and_names(), dist_model.parameters_and_names()):
        w1.set_data(w0)  # FIXME: seed does not work
        np.testing.assert_allclose(w0.value().asnumpy(), w1.value().asnumpy())

    # test forward
    non_dist_out = non_dist_model(**data)
    dist_out = dist_model(**data)

    np.testing.assert_allclose(non_dist_out.asnumpy(), dist_out.asnumpy(), atol=1e-5)
    print("Test 1 (Forward): Passed.")

    # test backward
    non_dist_mean_net = MeanNet(non_dist_model)
    dist_mean_net = MeanNet(dist_model)

    grad_fn = ops.value_and_grad(non_dist_mean_net, grad_position=None, weights=non_dist_mean_net.trainable_params())
    non_dist_loss, non_dist_grads = grad_fn(*data.values())

    grad_fn = ops.value_and_grad(dist_mean_net, grad_position=None, weights=dist_mean_net.trainable_params())
    dist_loss, dist_grads = grad_fn(*data.values())

    # take mean around different ranks
    sp_group = get_sequence_parallel_group()
    reduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=sp_group)
    num = get_group_size()
    syn_dist_grads = list()
    for x in dist_grads:
        syn_dist_grads.append(reduce(x) / num)

    np.testing.assert_allclose(non_dist_loss.asnumpy(), dist_loss.asnumpy(), atol=1e-5)

    for grad_0, grad_1 in zip(non_dist_grads, syn_dist_grads):
        np.testing.assert_allclose(grad_0.asnumpy(), grad_1.asnumpy(), atol=1e-5)
    print("Test 2 (Backward): Passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default=0, type=int, choices=[0, 1], help="Mode to test. (0: Graph Mode; 1: Pynative mode)"
    )
    args = parser.parse_args()
    run_model(mode=args.mode)
