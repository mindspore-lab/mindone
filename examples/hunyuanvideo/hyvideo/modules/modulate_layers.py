from typing import Callable

import mindspore as ms
from mindspore import nn


class ModulateDiT(nn.Cell):
    """Modulation layer for DiT."""
    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.act = act_layer()
        # Zero-initialize the modulation
        self.linear = nn.Dense(
            hidden_size, factor * hidden_size, has_bias=True, weight_init='zero', bias_init='zero') #, **factory_kwargs)
        
        self.dtype = dtype

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        # AMP: silu better be fp32, linear bf16. currently just use bf16 
        # TODO: in torch autocast, silu-exp is cast to fp32. 
        return self.linear(self.act(x.to(self.dtype)))

        # return self.linear(self.act(x.float()).to(self.dtype))

def modulate(x, shift=None, scale=None):
    """modulate by shift and scale

    Args:
        x (ms.Tensor): input tensor.
        shift (ms.Tensor, optional): shift tensor. Defaults to None.
        scale (ms.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        ms.Tensor: the output tensor after modulate.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_gate(x, gate=None, tanh=False):
    """

    Args:
        x (ms.Tensor): input tensor.
        gate (ms.Tensor, optional): gate tensor. Defaults to None.
        tanh (bool, optional): whether to use tanh function. Defaults to False.

    Returns:
        ms.Tensor: the output tensor after apply gate.
    """
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(1).tanh()
    else:
        return x * gate.unsqueeze(1)


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward
