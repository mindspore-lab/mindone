# reference to https://github.com/Stability-AI/generative-models
import logging
from abc import abstractmethod
from functools import partial
from typing import Iterable

from gm.modules.attention import SpatialTransformer
from gm.modules.diffusionmodules.util import (
    avg_pool_nd,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)
from gm.util import default, exists

import mindspore as ms
from mindspore import jit, nn, ops

_logger = logging.getLogger(__name__)


class TimestepBlock(nn.Cell):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def construct(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.SequentialCell, TimestepBlock):
    def construct(self, x, emb, context=None, **kwargs):
        for cell in self.cell_list:
            if isinstance(cell, TimestepBlock):
                x = cell(x, emb)
            elif isinstance(cell, SpatialTransformer):
                x = cell(x, context)
            else:
                x = cell(x)
        return x


class Upsample(nn.Cell):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, third_up=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.third_up = third_up
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding, pad_mode="pad")

    def construct(self, x):
        # assert x.shape[1] == self.channels
        if self.dims == 3:
            t_factor = 1 if not self.third_up else 2

            # x = ops.interpolate(x, size=(t_factor * x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest",)
            x = ops.ResizeNearestNeighborV2()(x, (t_factor * x.shape[2], x.shape[3] * 2, x.shape[4] * 2))
        else:
            # x = ops.interpolate(x, size=(x.shape[-2] * 2, x.shape[-1] * 2), mode="nearest")  # scale_factor=2., (not support with ms2.1)
            x = ops.ResizeNearestNeighborV2()(x, (x.shape[-2] * 2, x.shape[-1] * 2))
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Cell):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, third_down=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else ((1, 2, 2) if not third_down else (2, 2, 2))
        if use_conv:
            # disable building print
            # print(f"Building a Downsample layer with {dims} dims.")
            # print(
            #     f"  --> settings are: \n in-chn: {self.channels}, out-chn: {self.out_channels}, "
            #     f"kernel-size: 3, stride: {stride}, padding: {padding}"
            # )
            # if dims == 3:
            #     print(f"  --> Downsampling third axis (time): {third_down}")

            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding, pad_mode="pad")
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def construct(self, x):
        # assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            # MS doesn't support lists; MS requires padding for each side.
            padding = tuple(k // 2 for k in kernel_size for _ in range(2))
        else:
            padding = kernel_size // 2

        self.in_layers = nn.SequentialCell(
            [
                normalization(channels),
                nn.SiLU(),
                conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding, pad_mode="pad"),
            ]
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        if self.skip_t_emb:
            _logger.debug(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.SequentialCell(
                [
                    nn.SiLU(),
                    linear(
                        emb_channels,
                        self.emb_out_channels,
                    ),
                ]
            )

        self.out_layers = nn.SequentialCell(
            [
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding, pad_mode="pad")
                ),
            ]
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding, pad_mode="pad"
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def construct(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = ops.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = ops.chunk(emb_out, 2, axis=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                # emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                emb_out = emb_out.swapaxes(1, 2)  # (b, t, c, ...) -> (b, c, t, ...)
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


# TODO: Add Flash Attention Support
class AttentionBlock(nn.Cell):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def construct(self, x):
        b, c, _, _ = x.shape
        qkv = self.qkv(self.norm(x).reshape(b, c, -1))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(*x.shape)


class QKVAttentionLegacy(nn.Cell):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def construct(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, axis=1)

        # scale = 1 / math.sqrt(math.sqrt(ch))
        scale = 1 * ch**-0.25

        # weight = th.einsum(
        #     "bct,bcs->bts", q * scale, k * scale
        # )  # More stable with f16 than dividing afterwards
        weight = ops.BatchMatMul()((q * scale).transpose(0, 2, 1), (k * scale))  # (b, c, t) -> (b, t, c)  # (b, c, s)

        _weight_dtype = weight.dtype
        weight = ops.softmax(weight.astype(ms.float32), axis=-1).astype(_weight_dtype)

        # a = th.einsum("bts,bcs->bct", weight, v)
        a = ops.BatchMatMul()(weight, v.transpose(0, 2, 1)).transpose(  # (b, t, s)  # (b, c, s) -> (b, s, c)
            0, 2, 1
        )  # (b, t, c) -> (b, c, t)

        return a.reshape(bs, -1, length)


class QKVAttention(nn.Cell):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def construct(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        # assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, axis=1)
        scale = 1 * ch**-0.25  # 1 / math.sqrt(math.sqrt(ch))

        # weight = th.einsum(
        #     "bct,bcs->bts",
        #     (q * scale).view(bs * self.n_heads, ch, length),
        #     (k * scale).view(bs * self.n_heads, ch, length),
        # )  # More stable with f16 than dividing afterwards
        weight = ops.BatchMatMul()(
            (q * scale).view(bs * self.n_heads, ch, length).transpose(0, 2, 1),  # (b, c, t) -> (b, t, c)
            (k * scale).view(bs * self.n_heads, ch, length),  # (b, c, s)
        )

        _weight_dtype = weight.dtype
        weight = ops.softmax(weight.astype(ms.float32), axis=-1).astype(_weight_dtype)

        # a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        a = ops.BatchMatMul()(
            weight, v.reshape(bs * self.n_heads, ch, length).transpose(0, 2, 1)  # (b, t, s)  # (b, c, s) -> (b, s, c)
        ).transpose(
            0, 2, 1
        )  # (b, t, c) -> (b, c, t)

        return a.reshape(bs, -1, length)


class Timestep(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def construct(self, t):
        # TODO: Allow different return type, e.g. int32 -> float32
        return timestep_embedding(t, self.dim, dtype=t.dtype)


class UNetModel(nn.Cell):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        spatial_transformer_attn_type="vanilla",
        adm_in_channels=None,
        transformer_depth_middle=None,
        use_recompute=False,
    ):
        super().__init__()
        from omegaconf.listconfig import ListConfig

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        elif isinstance(transformer_depth, ListConfig):
            transformer_depth = list(transformer_depth)
        transformer_depth_middle = default(transformer_depth_middle, transformer_depth[-1])

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        # self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4

        self.time_embed = nn.SequentialCell(
            [
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            ]
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Dense(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.SequentialCell(
                    [
                        Timestep(model_channels),
                        nn.SequentialCell(
                            [
                                linear(model_channels, time_embed_dim),
                                nn.SiLU(),
                                linear(time_embed_dim, time_embed_dim),
                            ]
                        ),
                    ]
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.SequentialCell(
                    [
                        nn.SequentialCell(
                            [
                                linear(adm_in_channels, time_embed_dim),
                                nn.SiLU(),
                                linear(time_embed_dim, time_embed_dim),
                            ]
                        )
                    ]
                )
            else:
                raise ValueError()

        self.input_blocks = nn.CellList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1, pad_mode="pad"))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(  # always uses a self-attn
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                attn_type=spatial_transformer_attn_type,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.CellList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.SequentialCell(
            [
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1, pad_mode="pad")),
            ]
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.SequentialCell(
                [
                    normalization(ch),
                    conv_nd(dims, model_channels, n_embed, 1),
                    # nn.LogSoftmax(axis=1)  # change to cross_entropy and produce non-normalized logits
                ]
            )

        if use_recompute:
            self.recompute_strategy_v1()

    @jit
    def construct(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"
        hs, hs_idx = (), -1
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs += (h,)
            hs_idx += 1
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = ops.concat([h, hs[hs_idx]], axis=1)
            hs_idx -= 1
            h = module(h, emb, context)

        return self.out(h)

    def recompute_strategy_v1(self):
        # embed
        self.time_embed.recompute()
        self.label_emb.recompute()

        # input blocks
        self.input_blocks[4][0].recompute()  # 4
        self.input_blocks[5][0].recompute()  # 5
        self.input_blocks[7][0].recompute()  # 7
        self.input_blocks[8][0].recompute()  # 8

        # middle block
        self.middle_block[0].recompute()
        self.middle_block[1].recompute()

        # output blocks
        self.output_blocks[0][1].recompute()  # 0
        self.output_blocks[1][1].recompute()  # 1
        self.output_blocks[2][1].recompute()  # 2
        self.output_blocks[2][2].recompute()  # 2
        self.output_blocks[3][1].recompute()  # 3
        self.output_blocks[4][1].recompute()  # 4
        self.output_blocks[5][1].recompute()  # 5
        self.output_blocks[5][2].recompute()  # 5

        print("Turn on recompute with StrategyV1.")


class UNetModel_lora(UNetModel):
    def __init__(
        self, *, lora_dim=4, lora_alpha=None, lora_dropout=0.0, lora_merge_weights=True, only_save_lora=True, **kwargs
    ):
        use_recompute = kwargs.pop("use_recompute", False)
        super(UNetModel_lora, self).__init__(**kwargs)

        self.only_save_lora = only_save_lora

        from gm.modules.attention import CrossAttention, MemoryEfficientCrossAttention
        from gm.modules.lora import Dense as Dense_lora
        from gm.modules.lora import mark_only_lora_as_trainable

        for cell_name, cell in self.cells_and_names():
            if isinstance(cell, (CrossAttention, MemoryEfficientCrossAttention)):
                assert hasattr(cell, "to_q")
                query_dim, inner_dim = cell.to_q.in_channels, cell.to_q.out_channels
                cell.to_q = Dense_lora(
                    query_dim,
                    inner_dim,
                    has_bias=False,
                    r=lora_dim,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=lora_merge_weights,
                )
                _ = [_ for _ in map(partial(self._prefix_param, cell_name), cell.to_q.get_parameters())]

                assert hasattr(cell, "to_k")
                context_dim, inner_dim = cell.to_k.in_channels, cell.to_k.out_channels
                cell.to_k = Dense_lora(
                    context_dim,
                    inner_dim,
                    has_bias=False,
                    r=lora_dim,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=lora_merge_weights,
                )
                _ = [_ for _ in map(partial(self._prefix_param, cell_name), cell.to_k.get_parameters())]

                assert hasattr(cell, "to_v")
                context_dim, inner_dim = cell.to_v.in_channels, cell.to_v.out_channels
                cell.to_v = Dense_lora(
                    context_dim,
                    inner_dim,
                    has_bias=False,
                    r=lora_dim,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=lora_merge_weights,
                )
                _ = [_ for _ in map(partial(self._prefix_param, cell_name), cell.to_v.get_parameters())]

                assert hasattr(cell, "to_out")
                inner_dim, query_dim = cell.to_out[0].in_channels, cell.to_out[0].out_channels
                cell.to_out[0] = Dense_lora(
                    inner_dim,
                    query_dim,
                    r=lora_dim,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=lora_merge_weights,
                )
                _ = [_ for _ in map(partial(self._prefix_param, cell_name), cell.to_out.get_parameters())]

        mark_only_lora_as_trainable(self, bias="none")

        num_param = sum([p.size for _, p in self.parameters_and_names()])
        num_param_trainable = sum([p.size for p in self.trainable_params()])
        print(
            f"Unet_lora total params: {float(num_param) / 1e9}B, "
            f"trainable params: {float(num_param_trainable) / 1e6}M."
        )

        if use_recompute:
            self.middle_block.recompute()
            self.output_blocks.recompute()
            print("Turn on recompute, and the unet middle/output blocks will be recomputed.")

    @staticmethod
    def _prefix_param(prefix, param):
        if not param.name.startswith(prefix):
            param.name = f"{prefix}.{param.name}"


class UNetModelStage1(nn.Cell):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        spatial_transformer_attn_type="vanilla",
        adm_in_channels=None,
        transformer_depth_middle=None,
        use_recompute=False,
    ):
        super().__init__()
        from omegaconf.listconfig import ListConfig

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        elif isinstance(transformer_depth, ListConfig):
            transformer_depth = list(transformer_depth)
        transformer_depth_middle = default(transformer_depth_middle, transformer_depth[-1])

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        # self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4

        self.time_embed = nn.SequentialCell(
            [
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            ]
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Dense(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.SequentialCell(
                    [
                        Timestep(model_channels),
                        nn.SequentialCell(
                            [
                                linear(model_channels, time_embed_dim),
                                nn.SiLU(),
                                linear(time_embed_dim, time_embed_dim),
                            ]
                        ),
                    ]
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.SequentialCell(
                    [
                        nn.SequentialCell(
                            [
                                linear(adm_in_channels, time_embed_dim),
                                nn.SiLU(),
                                linear(time_embed_dim, time_embed_dim),
                            ]
                        )
                    ]
                )
            else:
                raise ValueError()

        self.input_blocks = nn.CellList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1, pad_mode="pad"))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(  # always uses a self-attn
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                attn_type=spatial_transformer_attn_type,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        if use_recompute:
            self.recompute_strategy_v1()

    @jit
    def construct(self, x, timesteps=None, context=None, y=None, **kwargs):
        hs, hs_idx = (), -1
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs += (h,)
            hs_idx += 1
        h = self.middle_block(h, emb, context)
        # for module in self.output_blocks:
        #     h = ops.concat([h, hs[hs_idx]], axis=1)
        #     hs_idx -= 1
        #     h = module(h, emb, context)
        #
        # return self.out(h)

        context = ops.stop_gradient(context)
        outs = (h, emb, context) + hs
        return outs

    def recompute_strategy_v1(self):
        # embed
        self.time_embed.recompute()
        self.label_emb.recompute()

        # input blocks
        self.input_blocks[4][0].recompute()  # 4
        self.input_blocks[5][0].recompute()  # 5
        self.input_blocks[7][0].recompute()  # 7
        self.input_blocks[8][0].recompute()  # 8

        # middle block
        self.middle_block[0].recompute()
        self.middle_block[1].recompute()

        print("Turn on recompute with StrategyV1.")


class UNetModelStage2(nn.Cell):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        spatial_transformer_attn_type="vanilla",
        adm_in_channels=None,
        transformer_depth_middle=None,
        use_recompute=False,
    ):
        super().__init__()
        from omegaconf.listconfig import ListConfig

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        elif isinstance(transformer_depth, ListConfig):
            transformer_depth = list(transformer_depth)
        transformer_depth_middle = default(transformer_depth_middle, transformer_depth[-1])

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        # self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4

        self.input_blocks = [
            None,
        ]
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    None,
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(None)
                self.input_blocks.append(layers)
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(None)
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = None
        self._feature_size += ch

        self.output_blocks = nn.CellList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.SequentialCell(
            [
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1, pad_mode="pad")),
            ]
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.SequentialCell(
                [
                    normalization(ch),
                    conv_nd(dims, model_channels, n_embed, 1),
                    # nn.LogSoftmax(axis=1)  # change to cross_entropy and produce non-normalized logits
                ]
            )

        if use_recompute:
            # self.recompute_strategy_v0()
            self.recompute_strategy_v1()

    @jit
    def construct(self, h, emb, context, *hs):
        # h, emb, context, hs[0:9]
        hs_idx = 8
        for module in self.output_blocks:
            h = ops.concat([h, hs[hs_idx]], axis=1)
            hs_idx -= 1
            h = module(h, emb, context)

        return self.out(h)

    def recompute_strategy_v1(self):
        # output blocks
        self.output_blocks[0][1].recompute()  # 0
        self.output_blocks[1][1].recompute()  # 1
        self.output_blocks[2][1].recompute()  # 2
        self.output_blocks[2][2].recompute()  # 2
        self.output_blocks[3][1].recompute()  # 3
        self.output_blocks[4][1].recompute()  # 4
        self.output_blocks[5][1].recompute()  # 5
        self.output_blocks[5][2].recompute()  # 5

        print("Turn on recompute with StrategyV1.")
