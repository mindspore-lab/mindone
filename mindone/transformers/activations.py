# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
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

import functools
import math
from collections import OrderedDict

from transformers.utils import logging

import mindspore as ms
from mindspore import Tensor, mint, nn

logger = logging.get_logger(__name__)


class GELUTanh(nn.Cell):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://huggingface.co/papers/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self, use_gelu_tanh_python: bool = False):
        super().__init__()
        if use_gelu_tanh_python:
            self.act = self._gelu_tanh_python
        else:
            self.act = functools.partial(mint.nn.functional.gelu, approximate="tanh")

    def _gelu_tanh_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + mint.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * mint.pow(input, 3.0))))

    def construct(self, input: Tensor) -> Tensor:
        return self.act(input)


class NewGELUActivation(nn.Cell):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
    """

    def construct(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (1.0 + mint.tanh(mint.sqrt(Tensor(2.0 / math.pi)) * (input + 0.044715 * mint.pow(input, 3.0))))
        ).to(input.dtype)


class GELUActivation(nn.Cell):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    mint.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * mint.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = mint.nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + mint.erf(input / math.sqrt(2.0)))

    def construct(self, input: Tensor) -> Tensor:
        return self.act(input)


class SiLUActivation(nn.Cell):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def construct(self, input: Tensor) -> Tensor:
        return mint.nn.functional.silu(input)


class FastGELUActivation(nn.Cell):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def construct(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + mint.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Cell):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def construct(self, input: Tensor) -> Tensor:
        return input * mint.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Cell):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://huggingface.co/papers/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    ops.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * ops.pow(x, 3)))). See https://huggingface.co/papers/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max
        self.gelu = get_activation("gelu")

    def construct(self, x: Tensor) -> Tensor:
        return mint.clip(self.gelu(x), self.min, self.max)


class AccurateGELUActivation(nn.Cell):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def construct(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1 + mint.tanh(self.precomputed_constant * (input + 0.044715 * mint.pow(input, 3))))


class MishActivation(nn.Cell):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://huggingface.co/papers/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def construct(self, input: Tensor) -> Tensor:
        return mint.nn.functional.mish(input)


class LinearActivation(nn.Cell):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def construct(self, input: Tensor) -> Tensor:
        return input


class LaplaceActivation(nn.Cell):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://huggingface.co/papers/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """

    def construct(self, input, mu=0.707107, sigma=0.282095):
        input = (input - mu).div(sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + mint.erf(input))


class ReLUSquaredActivation(nn.Cell):
    """
    Applies the relu^2 activation introduced in https://huggingface.co/papers/2109.08668v2
    """

    def construct(self, input):
        relu_applied = mint.nn.functional.relu(input)
        squared = mint.square(relu_applied)
        return squared


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


class XIELUActivation(nn.Cell):
    """
    Applies the xIELU activation function introduced in https://arxiv.org/abs/2411.13010

    If the user has installed the nickjbrowning/XIELU wheel, we import xIELU CUDA
    Otherwise, we emit a single warning and use xIELU Python
    """

    def __init__(
        self,
        alpha_p_init=0.8,
        alpha_n_init=0.8,
        beta=0.5,
        eps=-1e-6,
        dtype=ms.bfloat16,
        with_vector_loads=False,
    ):
        super().__init__()
        self.alpha_p = ms.Parameter(mint.log(mint.expm1(ms.tensor(alpha_p_init, dtype=dtype))).unsqueeze(0))
        self.alpha_n = ms.Parameter(mint.log(mint.expm1(ms.tensor(alpha_n_init - beta, dtype=dtype))).unsqueeze(0))
        self.register_buffer("beta", ms.tensor(beta, dtype=dtype))
        self.register_buffer("eps", ms.tensor(eps, dtype=dtype))
        self.with_vector_loads = with_vector_loads
        logger.warning_once(
            "CUDA-fused xIELU not available (%s) – falling back to a Python version.\n"
            "CUDA xIELU (experimental), refer to `https://github.com/nickjbrowning/XIELU`"
        )

    def _xielu_python(self, x: Tensor) -> Tensor:
        alpha_p = mint.nn.functional.softplus(self.alpha_p)
        alpha_n = self.beta + mint.nn.functional.softplus(self.alpha_n)
        return mint.where(
            x > 0,
            alpha_p * x * x + self.beta * x,
            (mint.expm1(mint.min(x, self.eps)) - x) * alpha_n + self.beta * x,
        )

    def construct(self, input: Tensor) -> Tensor:
        return self._xielu_python(input)


ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": GELUTanh,
    "gelu_python_tanh": (GELUTanh, {"use_gelu_tanh_python": True}),
    "gelu_accurate": AccurateGELUActivation,
    "laplace": LaplaceActivation,
    "leaky_relu": (nn.LeakyReLU, {"alpha": 0.01}),
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": mint.nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": mint.nn.ReLU6,
    "sigmoid": mint.nn.Sigmoid,
    "silu": SiLUActivation,
    "swish": mint.nn.SiLU,
    "tanh": mint.nn.Tanh,
    "prelu": mint.nn.PReLU,
    "xielu": XIELUActivation,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


# For backwards compatibility with: from activations import gelu_python
gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")
