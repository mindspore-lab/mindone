r"""MindSpore re-implementation adapted from the Lua code in ``https://github.com/bobbens/sketch_simplification''.
"""

from mindspore import nn

from ...utils.pt2ms import load_pt_weights_in_model

__all__ = [
    "SketchSimplification",
    "sketch_simplification_gan",
]


class SketchSimplification(nn.Cell):
    r"""NOTE:
    1. Input image should has only one gray channel.
    2. Input image size should be divisible by 8.
    3. Sketch in the input/output image is in dark color while background in light color.
    """

    def __init__(self, mean, std):
        assert isinstance(mean, float) and isinstance(std, float)
        super(SketchSimplification, self).__init__()
        self.mean = mean
        self.std = std

        # layers
        self.layers = nn.SequentialCell(
            nn.Conv2d(1, 48, 5, 2, padding=2, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(48, 128, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(1024, 512, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2dTranspose(256, 256, 4, 2, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2dTranspose(128, 128, 4, 2, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(128, 48, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2dTranspose(48, 48, 4, 2, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(48, 24, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2d(24, 1, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.Sigmoid(),
        )

    def construct(self, x):
        r"""x: [B, 1, H, W] within range [0, 1]. Sketch pixels in dark color."""
        x = (x - self.mean) / self.std
        return self.layers(x)


def sketch_simplification_gan(pretrained=False, ckpt_path=None):
    model = SketchSimplification(mean=0.9664114577640158, std=0.0858381272736797)
    if pretrained:
        load_pt_weights_in_model(model, ckpt_path)
    return model


if __name__ == "__main__":
    model = sketch_simplification_gan(pretrained=False)
