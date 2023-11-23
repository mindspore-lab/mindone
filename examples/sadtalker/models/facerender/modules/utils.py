import mindspore as ms
from mindspore import nn, ops
from mindspore import dtype as mstype

from mindspore.nn import BatchNorm2d, BatchNorm3d
from models.facerender.modules.spectralnorm import Conv2dNormalized
from models.facerender.modules.instancenorm import InstanceNorm2d


def einsum():
    """Mindspore implementation of `torch.einsum`
    """
    pass


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']
    coordinate_grid = make_coordinate_grid(spatial_size, mean.dtype)
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)

    for i, num in enumerate(repeats):
        coordinate_grid = coordinate_grid.repeat(num, axis=i)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = ops.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    d, h, w = spatial_size
    x = ops.arange(w).astype(type)
    y = ops.arange(h).astype(type)
    z = ops.arange(d).astype(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)

    # yy = y.view(1, -1, 1).repeat(d, 1, w)
    # xx = x.view(1, 1, -1).repeat(d, h, 1)
    # zz = z.view(-1, 1, 1).repeat(1, h, w)

    yy = y.view(1, -1, 1).repeat(d, axis=0).repeat(w, axis=2)
    xx = x.view(1, 1, -1).repeat(d, axis=0).repeat(h, axis=1)
    zz = z.view(-1, 1, 1).repeat(h, axis=1).repeat(w, axis=2)

    meshed = ops.cat([xx.unsqueeze(3), yy.unsqueeze(3), zz.unsqueeze(3)], 3)

    return meshed


class ResBlock3d(nn.Cell):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               pad_mode='pad', padding=padding, has_bias=True)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               pad_mode='pad', padding=padding, has_bias=True)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def construct(self, x):
        out = self.norm1(x)
        out = ops.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = ops.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Cell):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              pad_mode='pad', padding=padding, group=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def construct(self, x):
        out = ops.interpolate(x, scale_factor=2.0, mode='area')
        out = self.conv(out)
        out = self.norm(out)
        out = ops.relu(out)
        return out


class UpBlock3d(nn.Cell):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock3d, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              pad_mode='pad', padding=padding, group=groups, has_bias=True)
        self.norm = BatchNorm3d(out_features, affine=True)

    def construct(self, x):
        out = ops.interpolate(x, scale_factor=(1.0, 2.0, 2.0), mode='area')
        out = self.conv(out)
        out = self.norm(out)
        out = ops.relu(out)
        return out


class DownBlock2d(nn.Cell):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              pad_mode='pad', padding=padding, group=groups, has_bias=True)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    def construct(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = ops.relu(out)
        out = self.pool(out)
        return out


class DownBlock3d(nn.Cell):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock3d, self).__init__()
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups, stride=(1, 2, 2))
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              pad_mode='pad', padding=padding, group=groups, has_bias=True)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def construct(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = ops.relu(out)
        out = self.pool(out)
        return out


class Encoder(nn.Cell):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features,
                                               block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.CellList(down_blocks)

    def construct(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Cell):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * \
                min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(
                UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.CellList(up_blocks)
        # self.out_filters = block_expansion
        self.out_filters = block_expansion + in_features

        self.conv = nn.Conv3d(
            in_channels=self.out_filters,
            out_channels=self.out_filters,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
            has_bias=True
        )
        self.norm = BatchNorm3d(self.out_filters, affine=True)

    def construct(self, x):
        out = x.pop()
        # for up_block in self.up_blocks[:-1]:
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = ops.cat([out, skip], axis=1)
        # out = self.up_blocks[-1](out)
        out = self.conv(out)
        out = self.norm(out)
        out = ops.relu(out)
        return out


class Hourglass(nn.Cell):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(
            block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(
            block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def construct(self, x):
        return self.decoder(self.encoder(x))


class KPHourglass(nn.Cell):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, reshape_features, reshape_depth, num_blocks=3, max_features=256):
        super(KPHourglass, self).__init__()

        self.down_blocks = nn.SequentialCell()
        for i in range(num_blocks):
            self.down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                                min(max_features,
                                                    block_expansion * (2 ** (i + 1))),
                                                kernel_size=3, padding=1))

        in_filters = min(max_features, block_expansion * (2 ** num_blocks))
        self.conv = nn.Conv2d(in_channels=in_filters,
                              out_channels=reshape_features,
                              kernel_size=1,
                              has_bias=True
                              )

        self.up_blocks = nn.SequentialCell()
        for i in range(num_blocks):
            in_filters = min(max_features, block_expansion *
                             (2 ** (num_blocks - i)))
            out_filters = min(max_features, block_expansion *
                              (2 ** (num_blocks - i - 1)))
            self.up_blocks.append(
                UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.reshape_depth = reshape_depth
        self.out_filters = out_filters

    def construct(self, x):
        out = self.down_blocks(x)
        out = self.conv(out)
        bs, c, h, w = out.shape
        out = out.view(bs, c//self.reshape_depth, self.reshape_depth, h, w)
        out = self.up_blocks(out)

        return out


class SameBlock2d(nn.Cell):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1, lrelu=False):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, pad_mode='pad', padding=padding, group=groups, has_bias=True)
        self.norm = BatchNorm2d(out_features, affine=True)
        if lrelu:
            self.ac = nn.LeakyReLU(0.01)
        else:
            self.ac = nn.ReLU()

    def construct(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ac(out)
        return out


class SPADE(nn.Cell):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        # self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.param_free_norm = InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128

        self.mlp_shared = nn.SequentialCell(
            nn.Conv2d(label_nc, nhidden, kernel_size=3,
                      pad_mode='pad', padding=1, has_bias=True),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(
            nhidden, norm_nc, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.mlp_beta = nn.Conv2d(
            nhidden, norm_nc, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)

    def construct(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = ops.interpolate(segmap, size=x.shape[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class SPADEResnetBlock(nn.Cell):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se

        # create conv layers, apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = Conv2dNormalized(
                fin, fmiddle, kernel_size=3, pad_mode='pad', padding=dilation, dilation=dilation, has_bias=True)
            self.conv_1 = Conv2dNormalized(
                fmiddle, fout, kernel_size=3, pad_mode='pad', padding=dilation, dilation=dilation, has_bias=True)
            if self.learned_shortcut:
                self.conv_s = Conv2dNormalized(
                    fin, fout, kernel_size=1, has_bias=False)

        else:
            self.conv_0 = nn.Conv2d(
                fin, fmiddle, kernel_size=3, pad_mode='pad', padding=dilation, dilation=dilation, has_bias=True)
            self.conv_1 = nn.Conv2d(
                fmiddle, fout, kernel_size=3, pad_mode='pad', padding=dilation, dilation=dilation, has_bias=True)
            if self.learned_shortcut:
                self.conv_s = nn.Conv2d(
                    fin, fout, kernel_size=1, has_bias=False)

        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def construct(self, x, seg1):
        x_s = self.shortcut(x, seg1)

        normalized_0 = self.norm_0(x, seg1)
        dx = self.conv_0(self.actvn(normalized_0))

        normalized_1 = self.norm_1(dx, seg1)
        dx = self.conv_1(self.actvn(normalized_1))

        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return ops.leaky_relu(x, alpha=2e-1)


class AntiAliasInterpolation2d(nn.Cell):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = ops.Meshgrid()(
            tuple([ops.arange(size, dtype=mstype.float32) for size in kernel_size]))
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= ops.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / ops.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.shape)
        # kernel = kernel.repeat([channels, *[1] * (kernel.dim() - 1)])
        kernel = kernel.repeat(channels, axis=0)

        # self.register_buffer('weight', kernel) ## TODO: requires_grad=False!!!
        self.weight = ms.Parameter(kernel, requires_grad=False)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def construct(self, input):
        if self.scale == 1.0:
            return input

        out = ops.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = ops.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


class ResBottleneck(nn.Cell):
    def __init__(self, in_features, stride):
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features//4,
            kernel_size=1,
            has_bias=True
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_features//4,
            out_channels=in_features // 4,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
            stride=stride,
            has_bias=True
        )
        self.conv3 = nn.Conv2d(
            in_channels=in_features // 4,
            out_channels=in_features,
            kernel_size=1,
            has_bias=True
        )
        self.norm1 = BatchNorm2d(in_features//4, affine=True)
        self.norm2 = BatchNorm2d(in_features//4, affine=True)
        self.norm3 = BatchNorm2d(in_features, affine=True)

        self.stride = stride
        if self.stride != 1:
            self.skip = nn.Conv2d(
                in_channels=in_features,
                out_channels=in_features,
                kernel_size=1,
                stride=stride,
                has_bias=True
            )
            self.norm4 = BatchNorm2d(in_features, affine=True)

    def construct(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = ops.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = ops.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.stride != 1:
            x = self.skip(x)
            x = self.norm4(x)
        out += x
        out = ops.relu(out)
        return out
