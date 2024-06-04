from collections import OrderedDict
from typing import Optional, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype

from ._common import LayerNorm, QuickGELU


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, has_bias=False).to_float(self.dtype)
        self.bn1 = nn.BatchNorm2d(planes).to_float(self.dtype)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, pad_mode="pad", has_bias=False).to_float(self.dtype)
        self.bn2 = nn.BatchNorm2d(planes).to_float(self.dtype)
        self.relu2 = nn.ReLU()

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, has_bias=False).to_float(self.dtype)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion).to_float(self.dtype)
        self.relu3 = nn.ReLU()

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.SequentialCell(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, has_bias=False).to_float(
                                self.dtype
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion).to_float(self.dtype)),
                    ]
                )
            )

    def construct(self, x: Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Cell):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, dtype: mstype = ms.float32
    ):
        super().__init__()
        self.dtype = dtype
        self.positional_embedding = Parameter(
            ms.numpy.randn(spacial_dim**2 + 1, embed_dim, dtype=self.dtype) / embed_dim**0.5
        )
        self.k_proj = nn.Dense(embed_dim, embed_dim).to_float(dtype)
        self.q_proj = nn.Dense(embed_dim, embed_dim).to_float(dtype)
        self.v_proj = nn.Dense(embed_dim, embed_dim).to_float(dtype)
        self.c_proj = nn.Dense(embed_dim, output_dim or embed_dim).to_float(dtype)
        self.num_heads = num_heads

    def construct(self, x: Tensor):
        x = x.flatten(start_dim=2)
        x = ops.transpose(x, (2, 0, 1))  # NCHW -> (HW)NC
        x = ops.concat([x.mean(axis=0, keep_dims=True), x], axis=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :]  # (HW+1)NC
        x, _ = ops.nn_func.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=ops.concat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Cell):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=False
        ).to_float(self.dtype)
        self.bn1 = nn.BatchNorm2d(width // 2).to_float(self.dtype)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, pad_mode="pad", padding=1, has_bias=False
        ).to_float(self.dtype)
        self.bn2 = nn.BatchNorm2d().to_float(self.dtype)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, pad_mode="pad", padding=1, has_bias=False).to_float(
            self.dtype
        )
        self.bn3 = nn.BatchNorm2d(width).to_float(self.dtype)
        self.relu3 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim, dtype=self.dtype)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride, dtype=self.dtype)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, dtype=self.dtype))

        return nn.SequentialCell(*layers)

    def construct(self, x: Tensor):
        def stem(x: Tensor):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class MultiheadAttention(nn.Cell):
    def __init__(self, d_model: int, n_head: int, dtype: mstype):
        super(MultiheadAttention, self).__init__()

        self.num_heads = n_head
        self.head_dim = d_model // n_head

        self.scaling = self.head_dim**-0.5
        self.in_proj = nn.Dense(d_model, 3 * d_model).to_float(dtype)
        self.out_proj = nn.Dense(d_model, d_model).to_float(dtype)

    def construct(self, query: ms.Tensor, attn_mask: Optional[ms.Tensor] = None):
        r"""Construct

        Args:
            query (ms.Tensor): query of attention.
            attn_mask (Optional[ms.Tensor]): attention mask.

        Returns:
            attn_output (ms.Tensor): attention output.
        """
        len_tgt, batch_size, width = query.shape
        qkv = self.in_proj(query)
        qkv = ops.reshape(qkv, (len_tgt, batch_size, 3, width)).transpose((2, 0, 1, 3))

        att_q = qkv[0:1]
        att_q = ops.Squeeze(0)(att_q)
        att_q = att_q * self.scaling
        att_q = att_q.view(len_tgt, batch_size * self.num_heads, self.head_dim).transpose((1, 0, 2))

        att_k = qkv[1:2]
        att_k = ops.Squeeze(0)(att_k)
        att_k = att_k.view(-1, batch_size * self.num_heads, self.head_dim).transpose((1, 0, 2))

        att_v = qkv[2:3]
        att_v = ops.Squeeze(0)(att_v)
        att_v = att_v.view(-1, batch_size * self.num_heads, self.head_dim).transpose((1, 0, 2))

        if attn_mask is not None:
            attn_output_weights = attn_mask + ops.matmul(att_q, att_k.transpose((0, 2, 1)))
        else:
            attn_output_weights = ops.matmul(att_q, att_k.transpose((0, 2, 1)))
        attn_output_weights = ops.softmax(attn_output_weights, axis=-1)
        attn_output = ops.matmul(attn_output_weights, att_v)
        attn_output = ops.transpose(attn_output, (1, 0, 2))
        attn_output = attn_output.view(len_tgt, batch_size, width)
        attn_output = self.out_proj(attn_output)
        return attn_output


class ResidualAttentionBlock(nn.Cell):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: Tensor = None,
        epsilon: float = 1e-5,
        use_quick_gelu: bool = False,
        mlp_ratio: float = 4.0,
        dtype: mstype = ms.float32,
    ):
        super().__init__()

        self.dtype = dtype
        self.attn = MultiheadAttention(d_model, n_head, dtype)
        self.ln_1 = LayerNorm((d_model,), epsilon=epsilon)
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", nn.Dense(d_model, round(d_model * mlp_ratio)).to_float(self.dtype)),
                    ("gelu", QuickGELU().to_float(self.dtype) if use_quick_gelu else nn.GELU().to_float(self.dtype)),
                    ("c_proj", nn.Dense(round(d_model * mlp_ratio), d_model).to_float(self.dtype)),
                ]
            )
        )
        self.ln_2 = LayerNorm((d_model,), epsilon=epsilon)
        self.attn_mask = attn_mask

    def attention(self, x: Tensor):
        return self.attn(x, attn_mask=self.attn_mask)

    def construct(self, x: Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Cell):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: Tensor = None,
        epsilon: float = 1e-5,
        use_quick_gelu: bool = False,
        mlp_ratio: float = 4.0,
        dtype: mstype = ms.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(
            *[
                ResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask,
                    epsilon=epsilon,
                    use_quick_gelu=use_quick_gelu,
                    mlp_ratio=mlp_ratio,
                    dtype=self.dtype,
                )
                for _ in range(layers)
            ]
        )

    def construct(self, x: Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Cell):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        epsilon: float = 1e-5,
        use_quick_gelu: bool = False,
        mlp_ratio: float = 4.0,
        dtype: mstype = ms.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, pad_mode="pad", has_bias=False
        ).to_float(self.dtype)

        scale = width**-0.5
        self.class_embedding = Parameter(scale * ms.numpy.randn(width, dtype=ms.float32))
        self.positional_embedding = Parameter(
            scale * ms.numpy.randn((input_resolution // patch_size) ** 2 + 1, width, dtype=ms.float32)
        )
        self.ln_pre = LayerNorm((width,), epsilon=epsilon)

        self.transformer = Transformer(
            width, layers, heads, epsilon=epsilon, use_quick_gelu=use_quick_gelu, mlp_ratio=mlp_ratio, dtype=self.dtype
        )

        self.ln_post = LayerNorm((width,), epsilon=epsilon)
        self.proj = Parameter(scale * ms.numpy.randn(width, output_dim, dtype=ms.float32))

    def construct(self, x: Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = ops.concat(
            [self.class_embedding.to(self.dtype) + ops.zeros((x.shape[0], 1, x.shape[-1]), dtype=self.dtype), x], axis=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj.to(x.dtype)

        return x


class ImageEncoder(nn.Cell):
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        vision_head_width: int,
        epsilon=1e-5,
        use_quick_gelu=False,
        mlp_ratio: float = 4.0,
        dtype: mstype = ms.float32,
    ):
        super().__init__()

        self.dtype = dtype

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // vision_head_width
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                dtype=self.dtype,
            )
        else:
            vision_heads = vision_width // vision_head_width
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                epsilon=epsilon,
                use_quick_gelu=use_quick_gelu,
                mlp_ratio=mlp_ratio,
                dtype=self.dtype,
            )

    def encode_image(self, image: Tensor):
        return self.visual(image)

    def construct(self, image: Tensor):
        return self.encode_image(image)
