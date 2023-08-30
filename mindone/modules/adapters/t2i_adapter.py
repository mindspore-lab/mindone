from collections import OrderedDict
from typing import List, Optional

from mindspore import Parameter, Tensor, amp, load_checkpoint, load_param_into_net, nn, ops

from examples.stable_diffusion_v2.tools._common.clip.clip_modules import QuickGELU


def get_adapter(condition: str, checkpoint: Optional[str] = None, use_fp16: bool = False) -> nn.Cell:
    """
    Get condition-specific T2I-Adapter.

    Args:
        condition: Condition for adapter. Possible values are: "style", "color", "sketch", "canny", "other"
        checkpoint: Path to weights checkpoint.
        use_fp16: Use half-precision adapter.

    Returns:
        T2I-Adapter.
    """
    if condition == "style":
        adapter = StyleT2IAdapter(num_token=8)
    elif condition == "color":
        adapter = T2IAdapter(
            arch="light",
            in_channels=3,
            out_channels=[320, 640, 1280, 1280],
            rb_num=4,
        )
    else:
        in_channels = 1 if condition in ["sketch", "canny"] else 3
        adapter = T2IAdapter(
            arch="full",
            in_channels=in_channels,
            out_channels=[320, 640, 1280, 1280],
            rb_num=2,
            kernel_size=1,
        )

    if checkpoint:
        param_dict = load_checkpoint(checkpoint)
        param_not_load, _ = load_param_into_net(adapter, param_dict)
        if param_not_load:
            raise ValueError(
                f"Failed to load the following adapter parameters: {param_not_load}."
                f"Please check that the path to the checkpoint is correct."
            )

    if use_fp16:
        adapter = amp.auto_mixed_precision(adapter, "O3")

    return adapter


class ResidualBlock(nn.Cell):
    """
    A modified ResNet18's BasicBlock (no BatchNorm).

    Args:
        channels: Number of input and output channels.
        kernel_size: Second convolution kernel size (first is fixed at 3).
        padding: Second convolution padding size (first is fixed at 1).
    """

    def __init__(self, channels: int, kernel_size: int, padding: int = 1):
        super().__init__()
        self.block = nn.SequentialCell(
            [
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding, pad_mode="pad", has_bias=True),
            ]
        )

    def construct(self, x: Tensor) -> Tensor:
        return self.block(x) + x


class T2IAdapter(nn.Cell):
    """
    Text-to-Image Adapter described in the `T2I-Adapter <https://arxiv.org/abs/2302.08453>`__ paper. It incorporates
    visual cues (such as color, depth, sketch, segmentation, etc.) to support text prompts in image generation.

    Args:
        arch: Architecture of the adapter. Either `full` or `light`.
        in_channels: Number of input image channels. Either 1 or 3.
        out_channels: List of output channels. Default: [320, 640, 1280, 1280].
        rb_num: Number of residual blocks in each stage. Default: 2.
        kernel_size: Kernel size of the RB's second convolution (Refer to `ResidualBlock` for details). Default: 1.
    """

    def __init__(
        self, arch: str, in_channels, out_channels: Optional[List[int]] = None, rb_num: int = 2, kernel_size: int = 1
    ):
        super().__init__()
        assert arch in ["full", "light"], f"T2IAdapter architecture must be `full` or `light`, got {arch}"
        assert in_channels in [1, 3], f"Input image channels to T2IAdapter must be 1 or 3, got {in_channels}"
        out_channels = out_channels or [320, 640, 1280, 1280]
        in_channels *= 64  # PixelUnshuffle converts (C, H×8, W×8) to (C×8×8, H, W)

        self.unshuffle = nn.PixelUnshuffle(downscale_factor=8)

        self.body = nn.CellList()
        for i in range(len(out_channels)):
            block = []
            if i != 0:
                block.append(nn.AvgPool2d(kernel_size=2, stride=2))

            in_ch = in_channels if i == 0 else out_channels[i - 1]
            if arch == "full":
                block.extend(self._full_block(i, in_ch, out_channels[i], rb_num, kernel_size))
            else:
                block.extend(self._light_block(in_ch, out_channels[i], rb_num))

            self.body.append(nn.SequentialCell(block))

    @staticmethod
    def _full_block(i: int, in_channels: int, out_channels: int, rb_num: int, kernel_size: int) -> List[nn.Cell]:
        """
        Constructs stage block for the full adapter.
        """
        padding = kernel_size // 2
        layers = []

        if in_channels != out_channels:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3 if i == 0 else kernel_size,
                    stride=1,
                    padding=1 if i == 0 else padding,
                    pad_mode="pad",
                    has_bias=True,
                )
            )

        for i in range(rb_num):
            layers.append(ResidualBlock(out_channels, kernel_size=kernel_size, padding=padding))
        return layers

    @staticmethod
    def _light_block(in_channels, out_channels, rb_num) -> List[nn.Cell]:
        """
        Constructs stage block for the light adapter. Light block operates on 1/4 the number of channels.
        """
        inter_channels = out_channels // 4

        layers = [nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)]
        for _ in range(rb_num):
            layers.append(ResidualBlock(inter_channels, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True))
        return layers

    def construct(self, x: Tensor) -> List[nn.Cell]:
        x = self.unshuffle(x)

        features = []
        for block in self.body:
            x = block(x)
            features.append(x)

        return features


# TODO: Replace with CLIP's ResidualAttentionBlock?
class ResidualAttentionBlock(nn.Cell):
    """
    Residual Attention Block from the `CLIP <https://arxiv.org/abs/2103.00020>`__ paper.

    Args:
        d_model: Width of the block.
        n_head: Number of attention heads.
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm((d_model,), epsilon=1e-5)
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", nn.Dense(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Dense(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm((d_model,), epsilon=1e-5)

    def attention(self, x: Tensor) -> Tensor:
        return self.attn(x, x, x, need_weights=False)[0]

    def construct(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class StyleT2IAdapter(nn.Cell):
    """
    Style T2I-Adapter. Mimics `CLIP <https://arxiv.org/abs/2103.00020>`__'s Vision Transformer architecture.

    Args:
        width: Width of the encoded style embedding.
        context_dim: Output style context dimension.
        num_head: Number of attention heads for each attention layer.
        n_layers: Number of attention layers.
        num_token: Number of style tokens.
    """

    def __init__(
        self, width: int = 1024, context_dim: int = 768, num_head: int = 8, n_layers: int = 3, num_token: int = 4
    ):
        super().__init__()

        scale = width**-0.5
        self.transformer_layers = nn.SequentialCell([ResidualAttentionBlock(width, num_head) for _ in range(n_layers)])
        self.num_token = num_token
        self.style_embedding = Parameter(ops.randn(1, num_token, width) * scale)
        self.ln_pre = nn.LayerNorm((width,), epsilon=1e-5)
        self.ln_post = nn.LayerNorm((width,), epsilon=1e-5)
        self.proj = Parameter(scale * ops.randn(width, context_dim))

    def construct(self, x: Tensor) -> Tensor:
        style_embedding = self.style_embedding + ops.zeros((x.shape[0], self.num_token, self.style_embedding.shape[-1]))
        x = ops.cat([x, style_embedding], axis=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_layers(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, -self.num_token :, :])
        x = x @ self.proj  # batch matmul

        return x
