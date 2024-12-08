import mindspore as ms
from mindspore import nn

from .model_utils import CausalConv3d, GroupNormExtend, get_activation_fn


class ResBlock(nn.Cell):
    def __init__(
        self,
        in_channels,  # SCH: added
        filters,
        conv_fn,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
        num_groups=32,
        dtype=ms.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.activate = activation_fn()
        self.use_conv_shortcut = use_conv_shortcut

        # SCH: MAGVIT uses GroupNorm by default
        self.norm1 = GroupNormExtend(num_groups, in_channels)
        self.conv1 = conv_fn(
            in_channels,
            self.filters,
            kernel_size=(3, 3, 3),
            has_bias=False,
            dtype=dtype,
        )
        self.norm2 = GroupNormExtend(num_groups, self.filters)
        self.conv2 = conv_fn(
            self.filters,
            self.filters,
            kernel_size=(3, 3, 3),
            has_bias=False,
            dtype=dtype,
        )
        if in_channels != filters:
            if self.use_conv_shortcut:
                self.conv3 = conv_fn(
                    in_channels,
                    self.filters,
                    kernel_size=(3, 3, 3),
                    has_bias=False,
                    dtype=dtype,
                )
            else:
                self.conv3 = conv_fn(
                    in_channels,
                    self.filters,
                    kernel_size=(1, 1, 1),
                    has_bias=False,
                    dtype=dtype,
                )

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


class Encoder(nn.Cell):
    """Encoder Blocks. (magvit version)"""

    def __init__(
        self,
        config,
        dtype=ms.float32,
    ):
        super().__init__()
        self.filters = config.filters
        self.num_res_blocks = config.num_enc_res_blocks
        self.num_blocks = len(config.channel_multipliers)
        self.channel_multipliers = config.channel_multipliers
        self.temporal_downsample = config.temporal_downsample
        self.spatial_downsample = config.spatial_downsample
        self.num_groups = config.num_groups
        self.embedding_dim = config.embedding_dim

        self.activation_fn = get_activation_fn(config.activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
            dtype=dtype,
        )

        # first layer conv
        self.conv_in = self.conv_fn(
            config.channels,
            self.filters,
            kernel_size=(3, 3, 3),
            has_bias=False,
            dtype=dtype,
        )

        # ResBlocks and conv downsample
        self.block_res_blocks = nn.CellList([])
        self.conv_blocks = nn.CellList([])

        filters = self.filters
        prev_filters = filters  # record for in_channels
        for i in range(self.num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            block_items = nn.CellList([])
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # update in_channels
            self.block_res_blocks.append(block_items)

            if i < self.num_blocks - 1:
                if self.spatial_downsample[i]:
                    t_stride = 2 if self.temporal_downsample[i] else 1
                    s_stride = 2
                    self.conv_blocks.append(
                        self.conv_fn(
                            prev_filters,
                            filters,
                            kernel_size=(3, 3, 3),
                            strides=(t_stride, s_stride, s_stride),
                            dtype=dtype,
                        )
                    )
                else:
                    # if no t downsample, don't add since this does nothing for pipeline models
                    self.conv_blocks.append(nn.Identity())  # Identity

                prev_filters = filters  # update in_channels

        # last layer res block
        self.res_blocks = nn.CellList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(prev_filters, filters, **self.block_args))
            prev_filters = filters  # update in_channels

        # MAGVIT uses Group Normalization
        self.norm1 = GroupNormExtend(self.num_groups, prev_filters)

        self.conv2 = self.conv_fn(
            prev_filters,
            self.embedding_dim,
            kernel_size=(1, 1, 1),
            pad_mode="same",
            dtype=dtype,
        )

    def construct(self, x):
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
    """Decoder Blocks. (magvit version)"""

    def __init__(
        self,
        config,
        dtype=ms.float32,
    ):
        super().__init__()
        self.filters = config.filters
        self.num_res_blocks = config.num_dec_res_blocks
        self.num_blocks = len(config.channel_multipliers)
        self.channel_multipliers = config.channel_multipliers
        self.temporal_downsample = config.temporal_downsample
        self.spatial_downsample = config.spatial_downsample
        self.num_groups = config.num_groups
        self.embedding_dim = config.embedding_dim
        self.s_stride = 2

        self.activation_fn = get_activation_fn(config.activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
            dtype=dtype,
        )

        filters = self.filters * self.channel_multipliers[-1]
        prev_filters = filters

        # last conv
        self.conv1 = self.conv_fn(
            self.embedding_dim,
            filters,
            kernel_size=(3, 3, 3),
            has_bias=True,
            dtype=dtype,
        )

        # last layer res block
        self.res_blocks = nn.CellList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(filters, filters, **self.block_args))

        # ResBlocks and conv upsample
        self.block_res_blocks = nn.CellList([])
        self.num_blocks = len(self.channel_multipliers)
        self.conv_blocks = nn.CellList([])

        # reverse to keep track of the in_channels, but append also in a reverse direction
        for i in reversed(range(self.num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            # resblock handling
            block_items = nn.CellList([])
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # SCH: update in_channels
            self.block_res_blocks.insert(0, block_items)  # SCH: append in front

            # conv blocks with upsampling
            if i > 0:
                # SCH: T-Causal Conv 3x3x3, f -> (t_stride * 2 * 2) * f, depth to space t_stride x 2 x 2
                if self.spatial_downsample[i - 1]:
                    t_stride = 2 if self.temporal_downsample[i - 1] else 1
                    # SCH: T-Causal Conv 3x3x3, f -> (t_stride * 2 * 2) * f, depth to space t_stride x 2 x 2
                    self.conv_blocks.insert(
                        0,
                        self.conv_fn(
                            prev_filters,
                            prev_filters * t_stride * self.s_stride * self.s_stride,
                            kernel_size=(3, 3, 3),
                            dtype=dtype,
                        ),
                    )
                else:
                    self.conv_blocks.insert(
                        0,
                        nn.Identity(),
                    )

        self.norm1 = GroupNormExtend(self.num_groups, prev_filters)

        self.conv_out = self.conv_fn(filters, config.channels, 3, dtype=dtype)

    def construct(self, x):
        x = self.conv1(x)
        for k in range(self.num_res_blocks):
            x = self.res_blocks[k](x)
        for i in reversed(range(self.num_blocks)):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                x = self.conv_blocks[i - 1](x)
                b, c, t, h, w = x.shape
                x = x.reshape(b, -1, t * t_stride, h * self.s_stride, w * self.s_stride)

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv_out(x)
        return x
