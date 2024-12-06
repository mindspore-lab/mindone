from typing import Tuple, Union

from packaging import version

import mindspore as ms
from mindspore import nn, ops

from ..layers.operation_selector import get_split_op


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def pad_at_dim(t, pad, dim=-1):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return ops.pad(t, (*zeros, *pad), mode="constant")


class CausalConv3d(nn.Cell):
    # default values aligned to torch
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode="constant",
        strides=None,  # allow custom stride
        has_bias=True,
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
        assert pad_mode == "constant"

        dilation = kwargs.pop("dilation", 1)
        stride = strides[0] if strides is not None else kwargs.pop("stride", 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad

        # diff from torch: torch pad order whc. ms pad order chw. need to reverse
        # self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)  # torch pad

        stride = strides if strides is not None else (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.time_causal_padding = (time_pad, 0, height_pad, height_pad, width_pad, width_pad)
        # integarte pad in conv can be faster
        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            has_bias=has_bias,
            pad_mode="pad",
            padding=self.time_causal_padding,
            **kwargs,
        )

    def construct(self, x):
        # x: (B, C, T, H, W)
        # x = ops.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv(x)
        return x


if version.parse(ms.__version__) >= version.parse("2.3"):
    GroupNorm = nn.GroupNorm
else:
    # GroupNorm does not support 5D input before 2.3
    class GroupNorm(nn.GroupNorm):
        # GroupNorm supporting tensors with more than 4 dim
        def construct(self, x):
            x_shape = x.shape
            if x.ndim >= 5:
                x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
            y = super().construct(x)

            return y.view(x_shape)


class ResBlock(nn.Cell):
    def __init__(
        self,
        in_channels,  # SCH: added
        filters,
        conv_fn=CausalConv3d,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
        num_groups=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.activate = activation_fn()
        self.use_conv_shortcut = use_conv_shortcut

        # SCH: MAGVIT uses GroupNorm by default
        self.norm1 = GroupNorm(num_groups, in_channels)
        self.conv1 = conv_fn(in_channels, self.filters, kernel_size=(3, 3, 3), has_bias=False)
        self.norm2 = GroupNorm(num_groups, self.filters)
        self.conv2 = conv_fn(self.filters, self.filters, kernel_size=(3, 3, 3), has_bias=False)
        if in_channels != filters:
            if self.use_conv_shortcut:
                self.conv3 = conv_fn(in_channels, self.filters, kernel_size=(3, 3, 3), has_bias=False)
            else:
                self.conv3 = conv_fn(in_channels, self.filters, kernel_size=(1, 1, 1), has_bias=False)

    def construct(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.in_channels != self.filters:  # SCH: ResBlock X->Y
            residual = self.conv3(residual)
        return x + residual


def get_activation_fn(activation):
    if activation == "relu":
        activation_fn = nn.ReLU
    elif activation == "swish":
        activation_fn = nn.SiLU
    else:
        raise NotImplementedError
    return activation_fn


class Encoder(nn.Cell):
    """Encoder Blocks."""

    @ms.lazy_inline()
    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=512,  # num channels for latent vector
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim

        self.activation_fn = get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        # NOTE: don't switch to Conv2d, unless adjust the default value of padding mode and has_hias for mindspore
        self.conv_fn = CausalConv3d

        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        # first layer conv
        self.conv_in = self.conv_fn(
            in_out_channels,
            filters,
            kernel_size=(3, 3, 3),
            has_bias=False,
        )

        # ResBlocks and conv downsample
        # TODO: need auto_prefix?
        self.block_res_blocks = nn.CellList(auto_prefix=False)
        self.conv_blocks = nn.CellList(auto_prefix=False)

        filters = self.filters
        prev_filters = filters  # record for in_channels
        for i in range(self.num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            block_items = nn.CellList()
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # update in_channels
            self.block_res_blocks.append(block_items)

            if i < self.num_blocks - 1:
                if self.temporal_downsample[i]:
                    t_stride = 2 if self.temporal_downsample[i] else 1
                    s_stride = 1
                    self.conv_blocks.append(
                        self.conv_fn(
                            prev_filters, filters, kernel_size=(3, 3, 3), strides=(t_stride, s_stride, s_stride)
                        )
                    )
                    prev_filters = filters  # update in_channels
                else:
                    # if no t downsample, don't add since this does nothing for pipeline models
                    self.conv_blocks.append(nn.Identity())  # Identity
                    prev_filters = filters  # update in_channels

            self.block_res_blocks.update_parameters_name(prefix=self.param_prefix + "block_res_blocks.")
            self.conv_blocks.update_parameters_name(prefix=self.param_prefix + "conv_blocks.")

        # last layer res block
        self.res_blocks = nn.CellList(auto_prefix=False)
        for i in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(prev_filters, filters, **self.block_args))
            prev_filters = filters  # update in_channels

            self.res_blocks.update_parameters_name(prefix=self.param_prefix + "res_blocks.")

        # MAGVIT uses Group Normalization
        self.norm1 = GroupNorm(self.num_groups, prev_filters)

        self.conv2 = self.conv_fn(
            prev_filters, self.embedding_dim, kernel_size=(1, 1, 1)
        )  # NOTE: since it is 1x1x1 conv, there is no need to set padding=same

    def construct(self, x):
        # input x: (b, z, t, h, w), z = 4
        # output: (b, embeddim_dim, t//4, h, w)
        x = self.conv_in(x)

        for i in range(self.num_blocks):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i < self.num_blocks - 1:
                x = self.conv_blocks[i](x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Cell):
    """Decoder Blocks."""

    @ms.lazy_inline()
    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=512,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),  # FIXME: better name temporal_upsample
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim
        self.s_stride = 1

        self.activation_fn = get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        filters = self.filters * self.channel_multipliers[-1]
        prev_filters = filters

        # last conv
        self.conv1 = self.conv_fn(self.embedding_dim, filters, kernel_size=(3, 3, 3), has_bias=True)

        # last layer res block
        self.res_blocks = nn.CellList(auto_prefix=False)
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(filters, filters, **self.block_args))

            self.res_blocks.update_parameters_name(prefix=self.param_prefix + "res_blocks.")

        # ResBlocks and conv upsample
        self.block_res_blocks = nn.CellList()
        self.num_blocks = len(self.channel_multipliers)
        self.conv_blocks = nn.CellList()
        # reverse to keep track of the in_channels, but append also in a reverse direction
        for i in reversed(range(self.num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            # resblock handling
            block_items = nn.CellList()
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # SCH: update in_channels
            self.block_res_blocks.insert(0, block_items)  # SCH: append in front

            # conv blocks with upsampling
            if i > 0:
                if self.temporal_downsample[i - 1]:
                    t_stride = 2 if self.temporal_downsample[i - 1] else 1
                    # SCH: T-Causal Conv 3x3x3, f -> (t_stride * 2 * 2) * f, depth to space t_stride x 2 x 2
                    self.conv_blocks.insert(
                        0,
                        self.conv_fn(
                            prev_filters, prev_filters * t_stride * self.s_stride * self.s_stride, kernel_size=(3, 3, 3)
                        ),
                    )
                else:
                    self.conv_blocks.insert(
                        0,
                        nn.Identity(),
                    )

            self.block_res_blocks.update_parameters_name(prefix=self.param_prefix + "block_res_blocks.")
            self.conv_blocks.update_parameters_name(prefix=self.param_prefix + "conv_blocks.")

        self.norm1 = GroupNorm(self.num_groups, prev_filters)

        self.conv_out = self.conv_fn(filters, in_out_channels, 3)

    @staticmethod
    def rearrange(x, ts, hs=1, ws=1):
        # "B (C ts hs ws) T H W -> B C (T ts) (H hs) (W ws)",
        # TODO: due to hs, ws is fixed to 1, it's equivalent to:
        #       B (C ts) T H W -> B C (T ts) H W )
        """for hs ws != 1
        B, C_ts_hs_ws, T, H, W = x.shape
        C = C_ts_hs_ws // (ts * hs * ws)
        x = ops.reshape(x, (B, C, ts, hs, ws, T, H, W))
        x = ops.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = ops.reshape(x, (B, C, T * ts, H * hs, W * ws))
        """

        B, C_ts, T, H, W = x.shape
        C = C_ts // ts
        x = ops.reshape(x, (B, C, ts, T, H, W))
        # (B, C, ts, T, H, W) -> (B, C, T, ts, H, W)
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (B, C, T * ts, H, W))

        return x

    def construct(self, x):
        # input: (b, embedding_dim, t//4, h, w)
        # output: (b, z, t, h, w)

        x = self.conv1(x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)
        # for i in reversed(range(self.num_blocks)):
        for i in list(range(self.num_blocks))[::-1]:
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                x = self.conv_blocks[i - 1](x)
                x = self.rearrange(
                    x,
                    ts=t_stride,
                    # hs=self.s_stride,
                    # ws=self.s_stride,
                )

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv_out(x)
        return x


class VAE_Temporal(nn.Cell):
    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(True, True, False),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
        use_recompute=False,
        sample_deterministic=False,
    ):
        super().__init__()

        self.time_downsample_factor = 2 ** sum(temporal_downsample)
        # self.time_padding = self.time_downsample_factor - 1
        self.patch_size = (self.time_downsample_factor, 1, 1)
        self.out_channels = in_out_channels
        self.sample_deterministic = sample_deterministic

        # NOTE: following MAGVIT, conv in bias=False in encoder first conv
        self.encoder = Encoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim * 2,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
        )
        self.quant_conv = CausalConv3d(2 * latent_embed_dim, 2 * embed_dim, 1)

        self.post_quant_conv = CausalConv3d(embed_dim, latent_embed_dim, 1)
        self.decoder = Decoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
        )
        self.split = ops.Split(axis=1, output_num=2)
        self.stdnormal = ops.StandardNormal()
        self.split = get_split_op()

        if use_recompute:
            self.recompute(self.encoder)
            self.recompute(self.quant_conv)
            self.recompute(self.post_quant_conv)
            self.recompute(self.decoder)

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute()
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        else:
            b.add_flags(output_no_recompute=True)

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            if input_size[i] is None:
                lsize = None
            elif i == 0:
                time_padding = (
                    0
                    if (input_size[i] % self.time_downsample_factor == 0)
                    else self.time_downsample_factor - input_size[i] % self.time_downsample_factor
                )
                lsize = (input_size[i] + time_padding) // self.patch_size[i]
            else:
                lsize = input_size[i] // self.patch_size[i]
            latent_size.append(lsize)
        return latent_size

    def _encode(self, x):
        # x: B C T H W
        B, C, T, H, W = x.shape
        time_padding = (
            0
            if (x.shape[2] % self.time_downsample_factor == 0)
            else self.time_downsample_factor - x.shape[2] % self.time_downsample_factor
        )
        # x = pad_at_dim(x, (time_padding, 0), dim=2)

        # x_dtype = x.dtype
        # FIXME: bf16 not supported for ops.pad. use concat. equivalent to: ops.pad(x, (0, 0, 0, 0, time_padding, 0), mode="constant")
        pad_tensor = ops.zeros((B, C, time_padding, H, W), x.dtype)
        x = ops.concat((pad_tensor, x), axis=2)

        encoded_feature = self.encoder(x)
        moments = self.quant_conv(encoded_feature).to(x.dtype)
        mean, logvar = self.split(moments, moments.shape[1] // 2, 1)

        return mean, logvar

    def sample(self, mean, logvar):
        # sample z from latent distribution
        logvar = ops.clip_by_value(logvar, -30.0, 20.0)
        std = ops.exp(0.5 * logvar)
        z = mean + std * self.stdnormal(mean.shape)

        return z

    def encode(self, x):
        # embedding, get latent representation z
        posterior_mean, posterior_logvar = self._encode(x)
        if self.sample_deterministic:
            return posterior_mean
        z = self.sample(posterior_mean, posterior_logvar)

        return z

    def decode(self, z, num_frames=None):
        time_padding = (
            0
            if (num_frames % self.time_downsample_factor == 0)
            else self.time_downsample_factor - num_frames % self.time_downsample_factor
        )
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        x = x[:, :, time_padding:]
        return x

    def construct(self, x, sample_posterior=True):
        posterior_mean, posterior_logvar = self._encode(x)
        if sample_posterior:
            z = self.sample(posterior_mean, posterior_logvar)
        else:
            z = posterior_mean
        recon_video = self.decode(z, num_frames=x.shape[2])
        return recon_video, posterior_mean, posterior_logvar, z


def VAE_Temporal_SD(from_pretrained=None, **kwargs):
    model = VAE_Temporal(
        in_out_channels=4,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        **kwargs,
    )
    if from_pretrained is not None:
        # load_checkpoint(model, ckpt_path=from_pretrained)
        sd = ms.load_checkpoint(from_pretrained)
        pu, cu = ms.load_param_into_net(model, sd, strict_load=False)
        print(f"Net param not loaded : {pu}")
        print(f"Checkpoint param not loaded : {cu}")

    return model
