# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mindspore as ms
from mindspore import nn, ops


def sigmoid(x):
    """A numerically stable version of the logistic sigmoid function."""
    return ops.where(
        x >= 0.0,
        1.0 / (1.0 + ops.exp(-x)),  # For positive values
        ops.exp(x) / (1.0 + ops.exp(x)),  # For negative values
    )


class SiLU(nn.Cell):
    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return x * sigmoid(x)


class FP32SiLU(nn.Cell):
    r"""
    SiLU activation function with input upcasted to mindspore.float32.
    """

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x_dtype = x.dtype
        x = ops.silu(x.float())
        x = x.to(x_dtype)
        return x


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Cell:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Cell: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


class GELU(nn.Cell):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out, has_bias=bias)
        self.approximate = approximate

    def gelu(self, gate: ms.Tensor) -> ms.Tensor:
        return ops.gelu(gate, approximate=self.approximate)

    def construct(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GEGLU(nn.Cell):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2, has_bias=bias)

    def gelu(self, gate: ms.Tensor) -> ms.Tensor:
        return ops.gelu(gate)

    def construct(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, axis=-1)
        return hidden_states * self.gelu(gate)


class SwiGLU(nn.Cell):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function. It's similar to `GEGLU`
    but uses SiLU / Swish instead of GeLU.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2, has_bias=bias)
        self.activation = nn.SiLU()

    def construct(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, axis=-1)
        return hidden_states * self.activation(gate)


class ApproximateGELU(nn.Cell):
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out, has_bias=bias)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.proj(x)
        return x * ops.sigmoid(1.702 * x)
