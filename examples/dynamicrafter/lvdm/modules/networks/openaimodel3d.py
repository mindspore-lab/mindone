import logging
from abc import abstractmethod

from lvdm.modules.attention import SpatialTransformer, TemporalTransformer
from lvdm.modules.networks.util import (
    Identity,
    avg_pool_nd,
    conv_nd,
    linear,
    normalization,
    rearrange_in_gn5d_bs,
    rearrange_out_gn5d,
    timestep_embedding,
    zero_module,
)

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer

from mindone.utils.version_control import is_old_ms_version

_logger = logging.getLogger(__name__)


class TimestepBlock(nn.Cell):
    """
    Any module where construct() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def construct(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.SequentialCell, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def construct(self, x, emb, context=None, batch_size=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, batch_size=batch_size)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, TemporalTransformer):
                x = rearrange_in_gn5d_bs(x, batch_size)
                x = layer(x, context)
                x = rearrange_out_gn5d(x)
            else:
                x = layer(x)
        return x


class Upsample(nn.Cell):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=ms.float32):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, 3, padding=padding, has_bias=True, pad_mode="pad"
            ).to_float(dtype)

    def construct(self, x, emb=None, context=None, target_size=None):
        if target_size is None:
            if self.dims == 3:
                x = ops.ResizeNearestNeighbor((x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2))(x)
            else:
                x = ops.ResizeNearestNeighbor((x.shape[2] * 2, x.shape[3] * 2))(x)
        else:
            x = ops.ResizeNearestNeighbor(size=target_size)(x)

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

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=ms.float32):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding, has_bias=True, pad_mode="pad"
            ).to_float(dtype)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def construct(self, x, emb=None, context=None):
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
        dropout=1.0,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        dtype=ms.float32,
        use_temporal_conv=False,
        tempspatial_aware=False
        # norm_in_5d=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.ori_channels = channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down
        self.dtype = dtype
        self.use_temporal_conv = use_temporal_conv
        self.identity = Identity()
        self.split = ops.Split(1, 2)

        # self.in_layers_norm = normalization(channels, norm_in_5d=norm_in_5d)
        self.in_layers_norm = normalization(
            channels
        )  # TODO: this is group norm actually, wrong naming. but renaming requires update of ckpt param name or mapping dict.
        self.in_layers_silu = nn.SiLU().to_float(self.dtype)
        self.in_layers_conv = conv_nd(
            dims, channels, self.out_channels, 3, padding=1, has_bias=True, pad_mode="pad"
        ).to_float(self.dtype)

        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=self.dtype)
            self.x_upd = Upsample(channels, False, dims, dtype=self.dtype)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=self.dtype)
            self.x_upd = Downsample(channels, False, dims, dtype=self.dtype)
        else:
            self.h_upd = self.x_upd = self.identity

        self.emb_layers = nn.SequentialCell(
            nn.SiLU().to_float(self.dtype),
            linear(
                emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels, dtype=self.dtype
            ),
        )

        self.out_layers_norm = normalization(self.out_channels)
        self.out_layers_silu = nn.SiLU().to_float(self.dtype)

        if is_old_ms_version():
            self.out_layers_drop = nn.Dropout(keep_prob=self.dropout)
        else:
            self.out_layers_drop = nn.Dropout(p=1.0 - self.dropout)

        self.out_layers_conv = zero_module(
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, has_bias=True, pad_mode="pad").to_float(
                self.dtype
            )
        )

        if self.out_channels == channels:
            self.skip_connection = self.identity
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1, has_bias=True, pad_mode="pad"
            ).to_float(self.dtype)
        else:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 1, has_bias=True, pad_mode="pad"
            ).to_float(self.dtype)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                spatial_aware=tempspatial_aware,
            )

    def construct(self, x, emb, batch_size=None):
        """
        x: (b*f c h w)
        """
        if self.updown:
            h = self.in_layers_norm(x)
            h = self.in_layers_silu(h)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.in_layers_conv(h)
        else:
            h = self.in_layers_norm(x)
            h = self.in_layers_silu(h)
            h = self.in_layers_conv(h)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = ops.expand_dims(emb_out, -1)

        if self.use_scale_shift_norm:
            scale, shift = self.split(emb_out)
            h = self.out_layers_norm(h) * (1 + scale) + shift
            h = self.out_layers_silu(h)
            h = self.out_layers_drop(h)
            h = self.out_layers_conv(h)

        else:
            h = h + emb_out
            h = self.out_layers_norm(h)
            h = self.out_layers_silu(h)
            h = self.out_layers_drop(h)
            h = self.out_layers_conv(h)

        h = self.skip_connection(x) + h

        if self.use_temporal_conv and batch_size:
            h = rearrange_in_gn5d_bs(h, batch_size)
            h = self.temopral_conv(h)
            h = rearrange_out_gn5d(h)
        return h


# SiLU fp32 compute
class SiLU(nn.SiLU):
    def construct(self, input_x):
        dtype = input_x.dtype
        return super().construct(input_x.to(ms.float32)).to(dtype)


class TemporalConvBlock(nn.Cell):
    def __init__(self, in_dim, out_dim=None, dropout=0.0, spatial_aware=False, dtype=ms.float32):
        super(TemporalConvBlock, self).__init__()
        if out_dim is None:
            out_dim = in_dim  # int(1.5*in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dtype = dtype
        th_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 3, 1)
        th_padding_shape = (1, 1, 0, 0, 0, 0) if not spatial_aware else (1, 1, 1, 1, 0, 0)
        tw_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 1, 3)
        tw_padding_shape = (1, 1, 0, 0, 0, 0) if not spatial_aware else (1, 1, 0, 0, 1, 1)

        # conv layers
        self.conv1 = nn.SequentialCell(
            normalization(in_dim),
            SiLU(),
            nn.Conv3d(
                in_dim, out_dim, th_kernel_shape, pad_mode="pad", padding=th_padding_shape, has_bias=True
            ).to_float(ms.float16),
        )
        self.conv2 = nn.SequentialCell(
            normalization(out_dim),
            SiLU(),
            nn.Dropout(1 - dropout) if is_old_ms_version() else nn.Dropout(p=dropout),
            nn.Conv3d(
                out_dim, in_dim, tw_kernel_shape, pad_mode="pad", padding=tw_padding_shape, has_bias=True
            ).to_float(ms.float16),
        )
        self.conv3 = nn.SequentialCell(
            normalization(out_dim),
            SiLU(),
            nn.Dropout(1 - dropout) if is_old_ms_version() else nn.Dropout(p=dropout),
            nn.Conv3d(
                out_dim, in_dim, th_kernel_shape, pad_mode="pad", padding=th_padding_shape, has_bias=True
            ).to_float(ms.float16),
        )
        self.conv4 = nn.SequentialCell(
            normalization(out_dim),
            SiLU(),
            nn.Dropout(1 - dropout) if is_old_ms_version() else nn.Dropout(p=dropout),
            nn.Conv3d(
                out_dim, in_dim, tw_kernel_shape, pad_mode="pad", padding=tw_padding_shape, has_bias=True
            ).to_float(ms.float16),
        )

        # zero out the last layer params,so the conv block is identity
        self.conv4[-1].weight.set_data(initializer("zeros", self.conv4[-1].weight.shape, self.conv4[-1].weight.dtype))
        self.conv4[-1].bias.set_data(initializer("zeros", self.conv4[-1].bias.shape, self.conv4[-1].bias.dtype))

    def construct(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return identity + x


class QKVAttention(nn.Cell):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads


class QKVAttentionLegacy(nn.Cell):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads


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


class Timestep(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def construct(self, t):
        return timestep_embedding(t, self.dim)


class UNetModel(nn.Cell):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: in_channels in the input Tensor.
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
        context_dim=None,
        use_scale_shift_norm=False,
        resblock_updown=False,
        num_heads=-1,
        num_head_channels=-1,
        transformer_depth=1,
        use_linear=False,
        temporal_conv=False,
        tempspatial_aware=False,
        temporal_attention=True,
        use_relative_position=True,
        use_causal_attention=False,
        temporal_length=None,
        use_fp16=False,
        addition_attention=False,
        temporal_selfatt_only=True,
        image_cross_attention=False,
        image_cross_attention_scale_learnable=False,
        default_fs=4,
        fs_condition=False,
        enable_flash_attention=False,
    ):
        super(UNetModel, self).__init__()
        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"
        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = 1.0 - dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.temporal_attention = temporal_attention
        time_embed_dim = model_channels * 4
        self.dtype = ms.float16 if use_fp16 else ms.float32
        temporal_self_att_only = True
        self.addition_attention = addition_attention
        self.temporal_length = temporal_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        self.default_fs = default_fs
        self.fs_condition = fs_condition
        self.enable_flash_attention = enable_flash_attention

        # Time embedding blocks
        self.time_embed = nn.SequentialCell(
            linear(model_channels, time_embed_dim, dtype=self.dtype),
            nn.SiLU().to_float(self.dtype),
            linear(time_embed_dim, time_embed_dim, dtype=self.dtype),
        )
        if fs_condition:
            self.fps_embedding = nn.SequentialCell(
                linear(model_channels, time_embed_dim, dtype=self.dtype),
                nn.SiLU().to_float(self.dtype),
                linear(time_embed_dim, time_embed_dim, dtype=self.dtype),
            )
            self.fps_embedding[-1].weight.set_data(
                initializer("zeros", self.fps_embedding[-1].weight.shape, self.fps_embedding[-1].weight.dtype)
            )
            self.fps_embedding[-1].bias.set_data(
                initializer("zeros", self.fps_embedding[-1].bias.shape, self.fps_embedding[-1].bias.dtype)
            )

        # Input Block
        self.input_blocks = nn.CellList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1, has_bias=True, pad_mode="pad")
                )
            ]
        )
        if self.addition_attention:
            self.init_attn = TimestepEmbedSequential(
                TemporalTransformer(
                    model_channels,
                    n_heads=8,
                    d_head=num_head_channels,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    only_self_att=temporal_selfatt_only,
                    causal_attention=False,
                    relative_position=use_relative_position,
                    temporal_length=temporal_length,
                )
            )

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_linear=use_linear,
                            disable_self_attn=False,
                            video_length=temporal_length,
                            image_cross_attention=self.image_cross_attention,
                            image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
                            enable_flash_attention=self.enable_flash_attention,
                        )
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                use_linear=use_linear,
                                only_self_att=temporal_self_att_only,
                                causal_attention=use_causal_attention,
                                relative_position=use_relative_position,
                                temporal_length=temporal_length,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
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

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        layers = [
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                tempspatial_aware=tempspatial_aware,
                use_temporal_conv=temporal_conv,
            ),
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                use_linear=use_linear,
                disable_self_attn=False,
                video_length=temporal_length,
                image_cross_attention=self.image_cross_attention,
                image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
                enable_flash_attention=self.enable_flash_attention,
            ),
        ]
        if self.temporal_attention:
            layers.append(
                TemporalTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_linear=use_linear,
                    only_self_att=temporal_self_att_only,
                    causal_attention=use_causal_attention,
                    relative_position=use_relative_position,
                    temporal_length=temporal_length,
                )
            )
        layers.append(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                tempspatial_aware=tempspatial_aware,
                use_temporal_conv=temporal_conv,
            )
        )

        # Middle Block
        self.middle_block = TimestepEmbedSequential(*layers)

        # Output Block
        self.output_blocks = nn.CellList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_linear=use_linear,
                            disable_self_attn=False,
                            video_length=temporal_length,
                            image_cross_attention=self.image_cross_attention,
                            image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
                            enable_flash_attention=self.enable_flash_attention,
                        )
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                use_linear=use_linear,
                                only_self_att=temporal_self_att_only,
                                causal_attention=use_causal_attention,
                                relative_position=use_relative_position,
                                temporal_length=temporal_length,
                            )
                        )
                if level and i == num_res_blocks:
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

        self.out = nn.SequentialCell(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1, has_bias=True, pad_mode="pad")),
        )

    def construct(self, x, timesteps, context=None, features_adapter=None, fs=None, **kwargs):
        b, _, t, _, _ = x.shape
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).astype(x.dtype)
        emb = self.time_embed(t_emb)

        # repeat t times for context [(b t) 77 768] & time embedding
        # check if we use per-frame image conditioning
        _, l_context, _ = context.shape
        if l_context == 77 + t * 16:  # !!! HARD CODE here
            context_text, context_img = context[:, :77, :], context[:, 77:, :]
            context_text = context_text.repeat_interleave(repeats=t, dim=0)

            # context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
            b, tl, c = context_img.shape
            context_img = ops.reshape(context_img, (b * t, tl // t, c))  # (b*t l c)

            context = ops.cat([context_text, context_img], axis=1)
        else:
            context = context.repeat_interleave(repeats=t, dim=0)
        emb = emb.repeat_interleave(repeats=t, dim=0)

        # always in shape (b t) c h w, except for temporal layer
        x = rearrange_out_gn5d(x)

        # combine emb
        if self.fs_condition:
            if fs is None:
                fs = ms.Tensor([self.default_fs] * b, dtype=ms.int64)
            fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).astype(x.dtype)

            fs_embed = self.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed

        h = x.astype(self.dtype)
        adapter_idx = 0
        hs = []
        for id, module in enumerate(self.input_blocks):
            h = module(h, emb, context=context, batch_size=b)
            if id == 0 and self.addition_attention:
                h = self.init_attn(h, emb, context=context, batch_size=b)
            # plug-in adapter features
            if ((id + 1) % 3 == 0) and features_adapter is not None:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1
            hs.append(h)
        if features_adapter is not None:
            assert len(features_adapter) == adapter_idx, "Wrong features_adapter"

        h = self.middle_block(h, emb, context=context, batch_size=b)
        for i, module in enumerate(self.output_blocks):
            hs_pop = hs[-(i + 1)]
            h = ops.cat([h, hs_pop], axis=1)
            h = module(h, emb, context=context, batch_size=b)
        h = h.astype(x.dtype)
        y = self.out(h)

        # reshape back to (b c t h w)
        y = rearrange_in_gn5d_bs(y, b)
        return y
