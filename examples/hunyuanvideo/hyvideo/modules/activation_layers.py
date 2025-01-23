from mindspore import nn

class SiLU(nn.SiLU):
    # compute in FP32
    def construct(self, x):
        input_dtype = x.dtype
        out = super().construct(x.float())
        return out.to(input_dtype)

# class MySiLU 

def get_activation_layer(act_type):
    if act_type == "gelu":
        return lambda: nn.GELU(approximate=False)
    elif act_type == "gelu_tanh":
        return lambda: nn.GELU(approximate=True)
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
        # return SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")
