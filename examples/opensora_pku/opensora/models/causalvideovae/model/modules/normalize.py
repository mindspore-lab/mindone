import mindspore as ms
from mindspore import Parameter, nn, ops


# TODO: put them to modules/normalize.py
class GroupNormExtend(nn.GroupNorm):
    # GroupNorm supporting tensors with more than 4 dim
    def construct(self, x):
        x_shape = x.shape
        if x.ndim >= 5:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
        y = super().construct(x)
        return y.view(x_shape)


def Normalize(in_channels, num_groups=32, extend=True):
    if extend:
        return GroupNormExtend(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ActNorm(nn.Cell):
    def __init__(self, num_features, logdet=False, affine=True, allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = Parameter(ops.zeros(1, num_features, 1, 1))
        self.scale = Parameter(ops.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.initialized = Parameter(ops.zeros(1, dtype=ms.uint8), requires_grad=False, name="initialized")

    def initialize(self, input):
        flatten = input.permute(1, 0, 2, 3).view(input.shape[1], -1)
        mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
        std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)

        self.loc.set_data(-mean)
        self.scale.set_data(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized[0] == 0:
            self.initialize(input)  # stop_grads?
            self.initialized.set_data(ops.ones(1, dtype=ms.uint8))

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = ops.log(ops.abs(self.scale))
            logdet = height * width * ops.sum(log_abs)
            logdet = logdet * ops.ones(input.shape[0]).to(input.dtype)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h
