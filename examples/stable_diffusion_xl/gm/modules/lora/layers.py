# reference to https://github.com/microsoft/LoRA

import math

from mindspore import Tensor, nn, ops
from mindspore.common import initializer as init


class Identity(nn.Cell):
    def construct(self, x):
        return x


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        assert r > 0, f"LoRA layer rank dim must be greater than 0, but got {r}."
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = Identity()

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Dense(nn.Dense, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        lora_alpha: int = None,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        assert r > 0, f"expected lora rank greater than 0, but got {r}"
        lora_alpha = lora_alpha if lora_alpha is not None else r

        nn.Dense.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        # Actual trainable parameters
        # self.lora_A = Parameter(Tensor(np.zeros((r, in_features)), self.weight.dtype))
        # self.lora_B = Parameter(Tensor(np.zeros((out_features, r)), self.weight.dtype))
        self.lora_A = nn.Dense(in_features, r, has_bias=False)
        self.lora_B = nn.Dense(r, out_features, has_bias=False)

        self.scaling = self.lora_alpha / self.r
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A.weight.set_data(
                init.initializer(
                    init.HeUniform(negative_slope=math.sqrt(5)), self.lora_A.weight.shape, self.lora_A.weight.dtype
                )
            )
            self.lora_B.weight.set_data(init.initializer(0.0, self.lora_B.weight.shape, self.lora_B.weight.dtype))

    def set_train(self, mode: bool = True):
        nn.Dense.set_train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    ops.assign(
                        self.weight, self.weight - ops.matmul(self.lora_B.weight, self.lora_A.weight) * self.scaling
                    )
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    ops.assign(
                        self.weight, self.weight + ops.matmul(self.lora_B.weight, self.lora_A.weight) * self.scaling
                    )
                self.merged = True

    def _linear(self, x):
        x_shape = self.shape_op(x)
        if len(x_shape) != 2:
            x = self.reshape(x, (-1, x_shape[-1]))
        x = self.matmul(x, self.weight)
        if self.has_bias:
            x = self.bias_add(x, self.bias)
        if self.activation_flag:
            x = self.activation(x)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (ops.shape(x)[-1],)
            x = self.reshape(x, out_shape)
        return x

    def construct(self, x: Tensor):
        if not self.merged:
            result = self._linear(x)
            result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        else:
            result = self._linear(x)

        return result
