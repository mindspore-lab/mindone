from collections import OrderedDict
from typing import List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

from mindspore import Parameter, Tensor, amp
from mindspore import dtype as ms_dtype
from mindspore import load_checkpoint, load_param_into_net, nn, ops


def get_adapter(
    diffusion_model: Literal["sd", "sdxl"],
    condition: str,
    checkpoint: Optional[str] = None,
    use_fp16: bool = False,
    train: bool = False,
) -> nn.Cell:
    """
    Get condition-specific T2I-Adapter.

    Args:
        diffusion_model: Stable Diffusion model version. Either "sd" or "sdxl".
        condition: Condition for adapter. Possible values are: "style", "color", "sketch", "canny", "other".
        checkpoint: Path to weights checkpoint.
        use_fp16: Use half-precision adapter.
        train: Run adapter in train or inference mode.

    Returns:
        T2I-Adapter.
    """
    if diffusion_model == "sdxl" and condition in ["style", "color"]:
        raise ValueError(f"Condition '{condition}' is not yet supported for SDXL model.")

    if condition == "style":
        adapter = StyleT2IAdapter(num_token=8)
    elif condition == "color":
        adapter = T2IAdapter(
            "sd",
            arch="light",
            in_channels=3,
            out_channels=[320, 640, 1280, 1280],
            rb_num=4,
        )
    else:
        # SDXL always operates on 3-channel images
        in_channels = 1 if diffusion_model == "sd" and condition in ["sketch", "canny"] else 3
        adapter = T2IAdapter(
            diffusion_model,
            arch="full",
            in_channels=in_channels,  # NOQA
            out_channels=[320, 640, 1280, 1280],
            rb_num=2,
            kernel_size=1,
        )

    if checkpoint:
        param_dict = load_checkpoint(checkpoint)
        pnames = list(param_dict.keys())
        for pn in pnames:
            new_pn = pn.replace("body.1", "body.0.1").replace("body.2", "body.0.2").replace("body.3", "body.0.3")
            param_dict[new_pn] = param_dict.pop(pn)

        param_not_load, _ = load_param_into_net(adapter, param_dict)
        if param_not_load:
            raise ValueError(
                f"Failed to load the following adapter parameters: {param_not_load}."
                f"Please check that the path to the checkpoint is correct."
            )

    adapter.set_train(train)
    if use_fp16:
        adapter = amp.auto_mixed_precision(adapter, "O2")

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
        self,
        diffusion_model: Literal["sd", "sdxl"],
        arch: Literal["full", "light"],
        in_channels: Literal[1, 3],
        out_channels: Optional[List[int]] = None,
        rb_num: int = 2,
        kernel_size: int = 1,
    ):
        super().__init__()
        out_channels = out_channels or [320, 640, 1280, 1280]
        df_mult = 1 if diffusion_model == "sd" else 2  # SDXL operates at 2x scale

        in_channels *= 64 * (df_mult**2)  # PixelUnshuffle converts (C, H×df, W×df) to (C×df×df, H, W)
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=8 * df_mult)

        self.body = nn.CellList()
        for i in range(len(out_channels)):
            block = []

            # SD and SDXL have different downsample stages
            if (diffusion_model == "sd" and i != 0) or (diffusion_model == "sdxl" and i == 2):
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

        layers.extend([ResidualBlock(out_channels, kernel_size=kernel_size, padding=padding) for _ in range(rb_num)])
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

    def construct(self, x: Tensor) -> List[Tensor]:
        x = self.unshuffle(x)

        features = []
        for block in self.body:
            x = block(x)
            features.append(x)

        return features


# TODO: Replace with CLIP's QuickGELU
class QuickGELU(nn.Cell):
    """
    A quick approximation of the GELU activation function.
    """

    def construct(self, x: Tensor) -> Tensor:
        return x * ops.sigmoid(1.702 * x)


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


class CombinedAdapter(nn.Cell):
    """
    Combine individual T2I-Adapters to infer on multiple conditions by summing up their weighted outputs.
    Also can be used with a single Adapter if features scaling (weighting) is desired.

    Args:
        adapters: A list of T2I-Adapters to combine.
        weights: A list of weights for each adapter. The larger the weight, the more aligned the generated image
                 and condition will be, but the generated quality may be reduced.
        output_fp16: Whether the output should be float16. Default: True.

    Raises:
        AssertionError: If the number of adapters and weights do not match.
    """

    def __init__(self, adapters: List[nn.Cell], weights: List[float], output_fp16: bool = True):
        super().__init__()
        assert len(adapters) == len(
            weights
        ), f"Number of adapters ({len(adapters)}) and weights ({len(weights)}) should match"

        self._adapters = adapters
        self._weights = weights
        self._out_cast = ms_dtype.float16 if output_fp16 else ms_dtype.float32

        self._regular_ids, self._style_ids = [], []
        for i, adapter in enumerate(adapters):
            # if an adapter is with automatic mixed precision applied
            instance_type = type(adapter._backbone if hasattr(adapter, "_backbone") else adapter)
            if instance_type == T2IAdapter:
                self._regular_ids.append(i)
            elif instance_type == StyleT2IAdapter:
                self._style_ids.append(i)

    def construct(self, conds: List[Tensor]) -> Tuple[Union[List[Tensor], None], Union[Tensor, None]]:
        """
        Combined adapters inference.

        Args:
            conds: A list of tensors representing conditions.

        Returns:
            A tuple containing the feature map and the context (for style transfer).
        """
        feature_map = None
        feature_seq = None

        if self._regular_ids:
            i = self._regular_ids[0]
            feature_map = [feat * self._weights[i] for feat in self._adapters[i](conds[i])]
            for i in self._regular_ids[1:]:
                feature_map = [
                    feature_map[j] + feat * self._weights[i] for j, feat in enumerate(self._adapters[i](conds[i]))
                ]

            feature_map = [feat.astype(self._out_cast) for feat in feature_map]

        if self._style_ids:
            feature_seq = [self._adapters[i](conds[i]) * self._weights[i] for i in self._style_ids]
            feature_seq = ops.cat(feature_seq, axis=1).astype(self._out_cast)

        return feature_map, feature_seq
