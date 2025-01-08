from mindspore import nn

def get_activation_layer(act_type):
    if act_type == "gelu":
        return lambda: nn.GELU(approximate=False)
    elif act_type == "gelu_tanh":
        return lambda: nn.GELU(approximate=True)
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")
