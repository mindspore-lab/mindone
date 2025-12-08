# coding=utf-8
# Copyright 2024 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from https://github.com/baaivision/Emu3 to work with MindSpore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Emu3VisionVQ model """

import math
from typing import Optional, Tuple, Union

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.initializer import Constant, HeNormal, Uniform, initializer

from mindone.transformers.modeling_utils import MSPreTrainedModel

from .configuration_emu3visionvq import Emu3VisionVQConfig


class Emu3VisionVQActivation(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x: ms.Tensor):
        return x * ops.sigmoid(x)


class Emu3VisionVQUpsample(nn.Cell):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

    def construct(self, x: ms.Tensor):
        x = ops.interpolate(x, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)
        x = self.conv(x)
        return x


class Emu3VisionVQDownsample(nn.Cell):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, pad_mode="pad", padding=0, has_bias=True
        )
        padding = ((0, 0), (0, 0), (0, 1), (0, 1))
        self.pad_op = ops.Pad(padding)

    def construct(self, x: ms.Tensor):
        # pad = (0, 1, 0, 1)
        # x = ops.pad(x, pad, mode="constant", value=0) # Graph mode not support bfloat16
        x = self.pad_op(x)
        x = self.conv(x)
        return x


class Emu3VisionVQCausalConv3d(nn.Cell):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Union[int, Tuple[int, ...]] = (3, 1, 1),
        stride: Union[int, Tuple[int, ...]] = (1, 1, 1),
        conv3d_dtype=ms.float16,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        hw_pad = [k - s for k, s in zip(kernel_size[1:], stride[1:])]
        self.padding = tuple()
        padding = ((0, 0), (0, 0), (2, 0))
        for p in hw_pad:
            padding += ((p // 2 + p % 2, p // 2),)  # NOTE: add in ((a,b),) format, make sure padding in (N,2) shape
        self.pad_op = ops.Pad(padding)
        # for p in hw_pad[::-1]:
        #     self.padding += (p // 2 + p % 2, p // 2),
        # self.padding += (2, 0)

        if ms.__version__ >= "2.5":
            self.conv = mint.nn.Conv3d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=0,
                bias=True,
            ).to_float(conv3d_dtype)
        else:
            self.conv = nn.Conv3d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                pad_mode="valid",
                has_bias=True,
            ).to_float(conv3d_dtype)

    def construct(self, x: ms.Tensor):
        origin_dtype = x.dtype
        # x = ops.pad(x, self.padding, mode="constant", value=0)
        x = self.pad_op(x)
        x = self.conv(x)
        x = x.to(origin_dtype)
        return x


class Emu3VisionVQResnetTemporalBlock(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        conv3d_dtype=ms.float16,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        stride = (1, 1, 1)
        kernel_size = (3, 3, 3)

        self.norm1 = mint.nn.BatchNorm3d(in_channels)
        self.conv1 = Emu3VisionVQCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.norm2 = mint.nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = Emu3VisionVQCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.act = Emu3VisionVQActivation()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Emu3VisionVQCausalConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            else:
                if ms.__version__ >= "2.5":
                    self.nin_shortcut = mint.nn.Conv3d(
                        in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
                    ).to_float(conv3d_dtype)
                else:
                    self.nin_shortcut = nn.Conv3d(
                        in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
                    ).to_float(conv3d_dtype)

    def construct(self, x: ms.Tensor):
        origin_dtype = x.dtype
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
            x = x.to(origin_dtype)

        return x + h


class Emu3VisionVQSpatialNorm(nn.Cell):
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        norm_layer: nn.Cell = nn.GroupNorm,
        add_conv: bool = False,
        num_groups: int = 32,
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()
        self.norm_layer = norm_layer(
            num_channels=f_channels,
            num_groups=num_groups,
            eps=eps,
            affine=affine,
        )

        self.add_conv = add_conv
        if self.add_conv:
            self.conv = nn.Conv2d(
                zq_channels, zq_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
            )

        self.conv_y = nn.Conv2d(
            zq_channels, f_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
        )
        self.conv_b = nn.Conv2d(
            zq_channels, f_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
        )

    def construct(self, x: ms.Tensor, zq: ms.Tensor):
        zq = ops.interpolate(zq, size=x.shape[-2:], mode="nearest")

        if self.add_conv:
            zq = self.conv(zq)

        x = self.norm_layer(x)
        x = x * self.conv_y(zq) + self.conv_b(zq)
        return x


class Emu3VisionVQResnetBlock(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        zq_ch: Optional[int] = None,
        add_conv: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.zq_ch = zq_ch

        if zq_ch is None:
            norm_kwargs = dict(num_groups=32, eps=1e-6, affine=True)
            self.norm1 = nn.GroupNorm(num_channels=in_channels, **norm_kwargs)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, **norm_kwargs)
        else:
            self.norm1 = Emu3VisionVQSpatialNorm(in_channels, zq_ch, add_conv=add_conv)
            self.norm2 = Emu3VisionVQSpatialNorm(out_channels, zq_ch, add_conv=add_conv)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        self.act = Emu3VisionVQActivation()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
                )

    def construct(self, x: ms.Tensor, zq: Optional[ms.Tensor] = None):
        norm_args = tuple() if self.zq_ch is None else (zq,)

        h = self.norm1(x, *norm_args)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h, *norm_args)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Emu3VisionVQAttnBlock(nn.Cell):
    def __init__(self, in_channels: int, zq_ch: Optional[int] = None, add_conv: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.zq_ch = zq_ch

        if zq_ch is None:
            norm_kwargs = dict(num_groups=32, eps=1e-6, affine=True)
            self.norm = nn.GroupNorm(num_channels=in_channels, **norm_kwargs)
        else:
            self.norm = Emu3VisionVQSpatialNorm(in_channels, zq_ch, add_conv=add_conv)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
        )

    def construct(self, x: ms.Tensor, zq: Optional[ms.Tensor] = None):
        norm_args = tuple() if self.zq_ch is None else (zq,)

        nx = self.norm(x, *norm_args)
        q = self.q(nx)
        k = self.k(nx)
        v = self.v(nx)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        score = ops.bmm(q.permute(0, 2, 1), k)
        score = score / (c**0.5)
        score = ops.softmax(score, axis=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        v = ops.bmm(v, score.permute(0, 2, 1))
        v = v.reshape(b, c, h, w)

        v = self.proj_out(v)

        return x + v


class Emu3VisionVQTemporalUpsample(nn.Cell):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Tuple[int, ...] = (3, 3, 3),
        stride: Tuple[int, ...] = (1, 1, 1),
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = Emu3VisionVQCausalConv3d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
        )

    def construct(self, x: ms.Tensor):
        b, c, t, h, w = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(b, c * h * w, t)  # (b, c, h, w, t) => (b, c*h*w, t)
        x = ops.interpolate(x, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)
        x = x.view(b, c, h, w, x.shape[-1]).permute(0, 1, 4, 2, 3).contiguous()
        x = self.conv(x)
        return x


class Emu3VisionVQTemporalDownsample(nn.Cell):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Tuple[int, ...] = (4, 3, 3),
        stride: Tuple[int, ...] = (2, 1, 1),
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.conv = Emu3VisionVQCausalConv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
        )

    def construct(self, x: ms.Tensor):
        x = self.conv(x)
        return x


class Emu3VisionVQVectorQuantizer(nn.Cell):
    def __init__(self, config: Emu3VisionVQConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.codebook_size, config.embed_dim)
        self.embedding.embedding_table.set_data(
            initializer(
                Uniform(scale=1.0 / config.codebook_size),
                self.embedding.embedding_table.shape,
                self.embedding.embedding_table.dtype,
            )
        )

    def construct(self, x: ms.Tensor):
        # b t c h w -> b t h w c
        b, t, c, h, w = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (b, t, h, w, c)
        x_flattened = x.view(b * t * h * w, c)

        codebook = self.embedding.embedding_table

        # ops.einsum('bd,dn->bn', x_flattened, codebook.permute(1, 0))
        einsum_res = ops.matmul(x_flattened, codebook.permute(1, 0))
        d = ops.sum(x_flattened**2, dim=1, keepdim=True) + ops.sum(codebook**2, dim=1) - 2 * einsum_res

        indices = ops.argmin(d, axis=1)
        indices = indices.view(b, t, h, w)
        return indices


class Emu3VisionVQEncoder(nn.Cell):
    def __init__(self, config: Emu3VisionVQConfig):
        super().__init__()
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.in_channels = config.in_channels

        # downsampling
        self.conv_in = nn.Conv2d(
            self.in_channels, self.ch, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.down = nn.CellList()
        for i_level in range(self.num_resolutions):
            block = nn.CellList()
            attn = nn.CellList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    Emu3VisionVQResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                    )
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(Emu3VisionVQAttnBlock(block_in))

            down = nn.Cell()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Emu3VisionVQDownsample(block_in)

            self.down.append(down)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = Emu3VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )
        self.mid.attn_1 = Emu3VisionVQAttnBlock(block_in)
        self.mid.block_2 = Emu3VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )

        # end
        self.norm_out = nn.GroupNorm(num_channels=block_in, num_groups=32, eps=1e-6, affine=True)

        out_z_channels = 2 * config.z_channels if config.double_z else config.z_channels
        self.conv_out = nn.Conv2d(
            block_in, out_z_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        temporal_down_blocks = int(math.log2(config.temporal_downsample_factor))
        self.time_conv = nn.CellList()

        for i in range(temporal_down_blocks):
            conv = Emu3VisionVQTemporalDownsample(out_z_channels, out_z_channels)
            self.time_conv.append(conv)

        self.time_res_stack = nn.SequentialCell(
            *[
                Emu3VisionVQResnetTemporalBlock(
                    in_channels=out_z_channels,
                    out_channels=out_z_channels,
                    dropout=config.dropout,
                )
                for _ in range(self.num_res_blocks)
            ]
        )

        self.act = Emu3VisionVQActivation()

    def construct(self, x: ms.Tensor):
        t = x.shape[1]
        x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = self.act(h)

        h = self.conv_out(h)

        h_n = h.shape[0] // t
        h = h.reshape(h_n, t, *h.shape[1:])
        h = h.permute(0, 2, 1, 3, 4)

        for conv in self.time_conv:
            h = self.act(conv(h))

        h = self.time_res_stack(h)
        h = h.permute(0, 2, 1, 3, 4)

        return h


class Emu3VisionVQDecoder(nn.Cell):
    def __init__(self, config: Emu3VisionVQConfig):
        super().__init__()
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks

        # in_ch_mult = (1,) + tuple(config.ch_mult)
        zq_ch = config.embed_dim

        block_in = config.ch * config.ch_mult[-1]
        self.time_res_stack = nn.SequentialCell(
            *[
                Emu3VisionVQResnetTemporalBlock(
                    in_channels=config.z_channels,
                    out_channels=config.z_channels,
                    dropout=config.dropout,
                )
                for _ in range(config.num_res_blocks)
            ]
        )

        tempo_upsample_block_num = int(math.log2(config.temporal_downsample_factor))
        self.time_conv = nn.CellList()
        for i in range(tempo_upsample_block_num):
            conv = Emu3VisionVQTemporalUpsample(config.z_channels, config.z_channels)
            self.time_conv.append(conv)

        self.conv_in = nn.Conv2d(
            config.z_channels, block_in, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = Emu3VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
            zq_ch=zq_ch,
        )
        self.mid.attn_1 = Emu3VisionVQAttnBlock(block_in, zq_ch)
        self.mid.block_2 = Emu3VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
            zq_ch=zq_ch,
        )

        # upsampling
        self.up = nn.CellList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.CellList()
            attn = nn.CellList()
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    Emu3VisionVQResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                        zq_ch=zq_ch,
                    )
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(Emu3VisionVQAttnBlock(block_in, zq_ch))

            up = nn.Cell()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Emu3VisionVQUpsample(block_in)

            self.up.insert(0, up)

        self.act = Emu3VisionVQActivation()

        self.norm_out = Emu3VisionVQSpatialNorm(block_in, zq_ch)
        self.conv_out = nn.Conv2d(
            block_in, config.out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

    def construct(self, z: ms.Tensor, zq: ms.Tensor):
        z_zq = mint.cat((z, zq), dim=0)
        z_zq = z_zq.permute(0, 2, 1, 3, 4)
        z_zq = self.time_res_stack(z_zq)

        for conv in self.time_conv:
            z_zq = self.act(conv(z_zq))

        z_zq = z_zq.permute(0, 2, 1, 3, 4)

        h, zq = mint.chunk(z_zq, 2, dim=0)

        h = h.reshape(h.shape[0] * h.shape[1], *h.shape[2:])
        zq = zq.reshape(zq.shape[0] * zq.shape[1], *zq.shape[2:])

        h = self.conv_in(h)

        # middle
        h = self.mid.block_1(h, zq)
        h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, zq)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h, zq)
        h = self.act(h)
        h = self.conv_out(h)

        return h


class Emu3VisionVQPretrainedModel(MSPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Emu3VisionVQConfig
    base_model_prefix = "emuvideovq"
    main_input_name = "pixel_values"
    _no_split_modules = ["Emu3VisionVQResnetBlock", "Emu3VisionVQAttnBlock", "Emu3VisionVQResnetTemporalBlock"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)) or (
            ms.__version__ >= "2.5" and isinstance(module, mint.nn.Conv3d)
        ):
            module.weight.set_data(
                initializer(HeNormal(mode="fan_out", nonlinearity="relu"), module.weight.shape, module.weight.dtype)
            )
        elif isinstance(module, nn.Linear):
            weight = initializer(HeNormal(negative_slope=math.sqrt(5)), module.weight.shape, module.weight.dtype)
            module.weight.set_data(weight)
            if module.bias is not None:
                fan_in, _ = initializer._calculate_fan_in_and_fan_out(module.weight.shape)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                bias_weight = initializer(
                    Uniform(scale=bound), module.embedding_table.shape, module.embedding_table.dtype
                )
                module.bias.set_data(bias_weight)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            module.gamma.set_data(initializer(Constant(1), module.gamma.shape, module.gamma.dtype))
            module.beta.set_data(initializer(Constant(0), module.beta.shape, module.beta.dtype))
        elif isinstance(module, mint.nn.BatchNorm3d):
            module.weight.set_data(initializer(Constant(1), module.weight.shape, module.weight.dtype))
            module.bias.set_data(initializer(Constant(0), module.bias.shape, module.bias.dtype))


class Emu3VisionVQModel(Emu3VisionVQPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = Emu3VisionVQEncoder(config)
        self.decoder = Emu3VisionVQDecoder(config)
        self.quantize = Emu3VisionVQVectorQuantizer(config)

        self.quant_conv = Emu3VisionVQCausalConv3d(config.z_channels, config.embed_dim)
        self.post_quant_conv = Emu3VisionVQCausalConv3d(config.embed_dim, config.z_channels)

        self.spatial_scale_factor = 2 ** (len(config.ch_mult) - 1)

        self.post_init()

    def encode(self, x: ms.Tensor):
        ndim = x.ndim
        if ndim == 4:
            t = self.config.temporal_downsample_factor
            b, c, h, w = x.shape
            x = x.unsqueeze(1).tile((1, t, 1, 1, 1))
        elif ndim == 5:
            b, t, c, h, w = x.shape

        h = self.encoder(x)

        # b t c h w -> b c t h w
        h = h.permute(0, 2, 1, 3, 4)
        h = self.quant_conv(h)
        # b c t h w -> b t c h w
        h = h.permute(0, 2, 1, 3, 4)

        codes = self.quantize(h)

        if ndim == 4:
            codes = codes.squeeze(1)

        return codes

    def decode(self, x: ms.Tensor):
        ndim = x.ndim
        if ndim == 3:
            x = x.unsqueeze(1)

        b, t, h, w = x.shape
        quant = self.quantize.embedding(x.flatten(start_dim=0))
        c = quant.shape[-1]
        quant = quant.view(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        quant2 = self.post_quant_conv(quant)

        quant = quant.permute(0, 2, 1, 3, 4)
        quant2 = quant2.permute(0, 2, 1, 3, 4)

        video = self.decoder(quant2, quant)
        video = video.reshape(
            b,
            t * self.config.temporal_downsample_factor,
            self.config.out_channels,
            h * self.spatial_scale_factor,
            w * self.spatial_scale_factor,
        )
        if ndim == 3:
            return video[:, 0]
        return video
