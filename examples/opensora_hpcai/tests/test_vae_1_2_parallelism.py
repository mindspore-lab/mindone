import argparse

import numpy as np
from opensora.acceleration.parallel_states import create_parallel_group
from opensora.models.vae.vae import OpenSoraVAE_V1_2
from opensora.utils.amp import auto_mixed_precision
from opensora.utils.model_utils import WHITELIST_OPS

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
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
    x = ops.rand([1, 3, 34, 720, 1080], dtype=ms.float32)  # (B, C, T, H, W)
    return dict(x=x)


def get_vae_config(micro_batch_parallel=False, micro_frame_parallel=False):
    config = {
        "micro_batch_size": 1,
        "micro_frame_size": 17,
        "micro_batch_parallel": micro_batch_parallel,
        "micro_frame_parallel": micro_frame_parallel,
        "sample_deterministic": True,
    }
    return config


def run_model(mode: int = 0, model_dtype: ms.dtype = ms.float16):
    ms.set_context(mode=mode)
    ms.set_auto_parallel_context(enable_alltoall=True)
    init()

    # prepare data
    set_random_seed(1024)
    data = get_sample_data()

    # single model
    set_random_seed(1024)
    non_dist_model_cfg = get_vae_config()
    non_dist_model = OpenSoraVAE_V1_2(**non_dist_model_cfg)
    if model_dtype != ms.float32 or mode == 0:
        non_dist_model = auto_mixed_precision(
            non_dist_model,
            amp_level="O2",
            dtype=model_dtype,
            custom_fp32_cells=WHITELIST_OPS,
        )
    non_dist_model.set_train(False)

    # sequence parallel model
    create_parallel_group(get_group_size())
    set_random_seed(1024)
    dist_model_cfg = get_vae_config(micro_batch_parallel=True, micro_frame_parallel=True)
    dist_model = OpenSoraVAE_V1_2(**dist_model_cfg)
    if model_dtype != ms.float32 or mode == 0:
        dist_model = auto_mixed_precision(
            dist_model,
            amp_level="O2",
            dtype=model_dtype,
            custom_fp32_cells=WHITELIST_OPS,
        )
    dist_model.set_train(False)

    for (_, w0), (_, w1) in zip(non_dist_model.parameters_and_names(), dist_model.parameters_and_names()):
        w1.set_data(w0)  # FIXME: seed does not work
        np.testing.assert_allclose(w0.value().asnumpy(), w1.value().asnumpy())

    # test forward
    if mode == 0:
        non_dist_encode = ms.jit(non_dist_model.encode, compile_once=False)
        non_dist_out = non_dist_encode(**data)
        dist_encode = ms.jit(dist_model.encode, compile_once=False)
        dist_out = dist_encode(**data)
    else:
        non_dist_encode = non_dist_model.encode
        non_dist_out = non_dist_encode(**data)
        dist_encode = dist_model.encode
        dist_out = dist_encode(**data)

    np.testing.assert_allclose(non_dist_out.asnumpy(), dist_out.asnumpy(), atol=1e-5)
    print("Test 1 (Forward): Passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default=0, type=int, choices=[0, 1], help="Mode to test. (0: Graph Mode; 1: Pynative mode)"
    )
    args = parser.parse_args()
    run_model(mode=args.mode)
