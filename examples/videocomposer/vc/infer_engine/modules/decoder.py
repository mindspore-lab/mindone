import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class Decoder(nn.Cell):
    """A wrapper of performing the decoding from the latent frames to pixel frames.

    Args:
        autoencoder: The AutoEncoder. It should accept input with a 4-Dimensional (b, c, h, w) tensor
        scale_factor: The scale value of the input. Default: 0.18215
    """

    def __init__(self, autoencoder: nn.Cell, scale_factor: float = 0.18215) -> None:
        super().__init__()
        self.autoencoder = autoencoder.set_train(False)
        for _, param in self.autoencoder.parameters_and_names():
            param.requires_grad = False

        self.scale_factor = scale_factor

    def construct(self, x: Tensor) -> Tensor:
        x = 1.0 / self.scale_factor * x
        b, c, f, h, w = x.shape
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (b * f, c, h, w))
        chunk_size = min(16, x.shape[0])
        n = x.shape[0] // chunk_size

        if n > 1:
            xs = ops.chunk(x, n, axis=0)
            decode_data = []
            for x in xs:
                tmp = self.autoencoder.decode(x)
                decode_data.append(tmp)
            x = ops.concat(decode_data, axis=0)
        else:
            x = self.autoencoder.decode(x)

        _, c, h, w = x.shape
        x = ops.reshape(x, (b, f, c, h, w))
        x = ops.transpose(x, (0, 2, 1, 3, 4))  # b c f h w
        return x
