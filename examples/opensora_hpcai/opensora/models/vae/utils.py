import math
from typing import Tuple

import mindspore as ms
from mindspore import mint, nn


ACT2CLS = {
    "swish": mint.nn.SiLU,
    "silu": mint.nn.SiLU,
    "mish": mint.nn.Mish,
    "gelu": nn.GELU,    # TODO
    "relu": mint.nn.ReLU,
}

# mint version of mindone.diffusers.models.activations.get_activation
def get_activation(act_fn: str) -> nn.Cell:

    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2CLS mapping {list(ACT2CLS.keys())}")


def ceil_to_divisible(n: int, dividend: int) -> int:
    return math.ceil(dividend / (dividend // n))


def get_conv3d_n_chunks(numel: int, n_channels: int, numel_limit: int):
    n_chunks = math.ceil(numel / numel_limit)
    n_chunks = ceil_to_divisible(n_chunks, n_channels)
    return n_chunks


class ChannelChunkConv3d(mint.nn.Conv3d):
    CONV3D_NUMEL_LIMIT = 2**31

    def _get_output_numel(self, input_shape: Tuple) -> int:
        numel = self.out_channels
        if len(input_shape) == 5:
            numel *= input_shape[0]
        for i, d in enumerate(input_shape[-3:]):
            d_out = math.floor(
                (d + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) / self.stride[i] + 1
            )
            numel *= d_out
        return numel

    def _get_n_chunks(self, numel: int, n_channels: int):
        n_chunks = math.ceil(numel / ChannelChunkConv3d.CONV3D_NUMEL_LIMIT)
        n_chunks = ceil_to_divisible(n_chunks, n_channels)
        return n_chunks

    def construct(self, input: ms.tensor) -> ms.tensor:
        if input.numel() // input.shape[0] < ChannelChunkConv3d.CONV3D_NUMEL_LIMIT:
            return super().construct(input)
        n_in_chunks = self._get_n_chunks(input.numel(), self.in_channels)
        n_out_chunks = self._get_n_chunks(self._get_output_numel(input.shape), self.out_channels)
        if n_in_chunks == 1 and n_out_chunks == 1:
            return super().construct(input)
        outputs = []
        input_shards = input.chunk(n_in_chunks, dim=1)
        for weight, bias in zip(self.weight.chunk(n_out_chunks), self.bias.chunk(n_out_chunks)):
            weight_shards = weight.chunk(n_in_chunks, dim=1)
            o = None
            for x, w in zip(input_shards, weight_shards):
                if o is None:
                    o = mint.nn.functional.conv3d(x, w, bias, self.stride, self.padding, self.dilation, self.groups)
                else:
                    o += mint.nn.functional.conv3d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
            outputs.append(o)
        return mint.cat(outputs, dim=1)
