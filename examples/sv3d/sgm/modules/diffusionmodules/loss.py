from typing import List, Optional, Union

from omegaconf import ListConfig
from sgm.util import append_dims

import mindspore as ms
from mindspore import nn, ops


class StandardDiffusionLoss(nn.Cell):
    def __init__(
        self,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
        keep_loss_fp32=True,
    ):
        super().__init__()

        assert type in ["l2", "l1"]
        self.type = type
        self.offset_noise_level = offset_noise_level
        self.keep_loss_fp32 = keep_loss_fp32

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def get_noise_input(self, pred, noise, sigmas):
        input = pred  # init a tensor in this node, for graph mode?
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                ops.randn(input.shape[0], dtype=input.dtype), input.ndim
            )
        noised_input = input + noise * append_dims(sigmas, input.ndim)
        return noised_input

    def construct(self, pred, target, w):
        if self.keep_loss_fp32:
            pred = ops.cast(pred, ms.float32)
            target = ops.cast(target, ms.float32)
        dtype = pred.dtype

        if self.type == "l2":
            loss = ops.mean((w * (pred - target) ** 2).reshape(target.shape[0], -1), 1).astype(dtype)
        elif self.type == "l1":
            loss = ops.mean((w * (pred - target).abs()).reshape(target.shape[0], -1), 1).astype(dtype)
        else:
            loss = 0.0
        return loss


class StandardDiffusionLoss2(StandardDiffusionLoss):
    def get_noise_input(self, pred, noise, sigmas):
        input = pred
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                ops.randn(input.shape[0], input.shape[1], dtype=input.dtype), input.ndim
            )
        noised_input = input + noise * append_dims(sigmas, input.ndim)
        return noised_input
