# Adapted from https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V to work with MindSpore.
from typing import Callable

import mindspore as ms
from mindspore import mint, nn


class ModulateDiT(nn.Cell):
    """Modulation layer for DiT."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable,
        dtype=None,
    ):
        # factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.act = act_layer()
        # Zero-initialize the modulation
        self.linear = mint.nn.Linear(
            hidden_size, factor * hidden_size, bias=True, weight_init="zero", bias_init="zero"
        )  # , **factory_kwargs)

        self.dtype = dtype

    def construct(self, x: ms.Tensor, condition_type=None, token_replace_vec=None) -> ms.Tensor:
        # AMP: silu better be fp32, linear bf16. currently just use bf16
        # TODO: in torch autocast, silu-exp is cast to fp32.
        x_out = self.linear(self.act(x))

        if condition_type == "token_replace":
            x_token_replace_out = self.linear(self.act(token_replace_vec))
            return x_out, x_token_replace_out
        else:
            return x_out

        # return self.linear(self.act(x.float()).to(self.dtype))


def modulate(x, shift=None, scale=None, condition_type=None, tr_shift=None, tr_scale=None, frist_frame_token_num=None):
    """modulate by shift and scale

    Args:
        x (ms.Tensor): input tensor.
        shift (ms.Tensor, optional): shift tensor. Defaults to None.
        scale (ms.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        ms.Tensor: the output tensor after modulate.
    """
    if condition_type == "token_replace":
        x_zero = x[:, :frist_frame_token_num] * (1 + tr_scale.unsqueeze(1)) + tr_shift.unsqueeze(1)
        x_orig = x[:, frist_frame_token_num:] * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = mint.concat((x_zero, x_orig), dim=1)
        return x
    else:
        if scale is None and shift is None:
            return x
        elif shift is None:
            return x * (1 + scale.unsqueeze(1))
        elif scale is None:
            return x + shift.unsqueeze(1)
        else:
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_gate(x, gate=None, tanh=False, condition_type=None, tr_gate=None, frist_frame_token_num=None):
    """

    Args:
        x (ms.Tensor): input tensor.
        gate (ms.Tensor, optional): gate tensor. Defaults to None.
        tanh (bool, optional): whether to use tanh function. Defaults to False.

    Returns:
        ms.Tensor: the output tensor after apply gate.
    """
    if condition_type == "token_replace":
        if gate is None:
            return x
        if tanh:
            x_zero = mint.tanh(x[:, :frist_frame_token_num] * tr_gate.unsqueeze(1))
            x_orig = mint.tanh(x[:, frist_frame_token_num:] * gate.unsqueeze(1))
            x = mint.concat((x_zero, x_orig), dim=1)
            return x
        else:
            x_zero = x[:, :frist_frame_token_num] * tr_gate.unsqueeze(1)
            x_orig = x[:, frist_frame_token_num:] * gate.unsqueeze(1)
            x = mint.concat((x_zero, x_orig), dim=1)
            return x
    else:
        if gate is None:
            return x
        if tanh:
            return mint.tanh(x * gate.unsqueeze(1))
        else:
            return x * gate.unsqueeze(1)


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward
