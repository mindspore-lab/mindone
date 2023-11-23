import mindspore as ms
from mindspore import nn, ops

from mindspore.nn import BatchNorm2d
from models.facerender.modules.utils import KPHourglass, make_coordinate_grid, AntiAliasInterpolation2d, ResBottleneck


class KPDetector(nn.Cell):
    """
    Detecting canonical keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, reshape_channel, reshape_depth,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1, single_jacobian_map=False):
        super(KPDetector, self).__init__()

        self.predictor = KPHourglass(
            block_expansion,
            in_features=image_channel,
            max_features=max_features,
            reshape_features=reshape_channel,
            reshape_depth=reshape_depth,
            num_blocks=num_blocks
        )

        self.kp = nn.Conv3d(
            in_channels=self.predictor.out_filters,
            out_channels=num_kp,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
            has_bias=True
        )

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv3d(
                in_channels=self.predictor.out_filters,
                out_channels=9 * self.num_jacobian_maps,
                kernel_size=3,
                pad_mode='pad',
                padding=1,
                has_bias=True
            )
            '''
            initial as:
            [[1 0 0]
             [0 1 0]
             [0 0 1]]
            '''
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(ms.tensor(
                [1, 0, 0, 0, 1, 0, 0, 0, 1] * self.num_jacobian_maps, dtype=ms.float32))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(
                image_channel, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(
            shape[2:], heatmap.dtype).unsqueeze(0).unsqueeze(0)
        value = (heatmap * grid).sum(axis=(2, 3, 4))
        kp = {'value': value}

        return kp

    def construct(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = ops.softmax(heatmap / self.temperature, axis=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 9, final_shape[2],
                                                final_shape[3], final_shape[4])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 9, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(
                jacobian.shape[0], jacobian.shape[1], 3, 3)
            out['jacobian'] = jacobian

        return out


class HEEstimator(nn.Cell):
    """
    Estimating head pose and expression.
    """

    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, num_bins=66, estimate_jacobian=True):
        super(HEEstimator, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=image_channel,
            out_channels=block_expansion,
            kernel_size=7,
            pad_mode='pad',
            padding=3,
            stride=2,
            has_bias=True
        )
        self.norm1 = BatchNorm2d(block_expansion, affine=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, pad_mode='pad', padding=1)

        self.conv2 = nn.Conv2d(
            in_channels=block_expansion,
            out_channels=256,
            kernel_size=1,
            has_bias=True
        )
        self.norm2 = BatchNorm2d(256, affine=True)

        block1 = nn.SequentialCell()
        for i in range(3):
            block1.append(ResBottleneck(in_features=256, stride=1))
        self.block1 = block1

        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=1, has_bias=True)
        self.norm3 = BatchNorm2d(512, affine=True)
        self.block2 = ResBottleneck(in_features=512, stride=2)

        block3 = nn.SequentialCell()
        for i in range(3):
            block3.append(ResBottleneck(in_features=512, stride=1))
        self.block3 = block3

        self.conv4 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=1, has_bias=True)
        self.norm4 = BatchNorm2d(1024, affine=True)
        self.block4 = ResBottleneck(in_features=1024, stride=2)

        block5 = nn.SequentialCell()
        for i in range(5):
            block5.append(ResBottleneck(in_features=1024, stride=1))
        self.block5 = block5

        self.conv5 = nn.Conv2d(
            in_channels=1024, out_channels=2048, kernel_size=1, has_bias=True)
        self.norm5 = BatchNorm2d(2048, affine=True)
        self.block6 = ResBottleneck(in_features=2048, stride=2)

        block7 = nn.SequentialCell()
        for i in range(2):
            block7.append(ResBottleneck(in_features=2048, stride=1))
        self.block7 = block7

        self.fc_roll = nn.Dense(2048, num_bins)
        self.fc_pitch = nn.Dense(2048, num_bins)
        self.fc_yaw = nn.Dense(2048, num_bins)
        self.fc_t = nn.Dense(2048, 3)
        self.fc_exp = nn.Dense(2048, 3*num_kp)

    def construct(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = ops.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = ops.relu(out)

        out = self.block1(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = ops.relu(out)
        out = self.block2(out)

        out = self.block3(out)

        out = self.conv4(out)
        out = self.norm4(out)
        out = ops.relu(out)
        out = self.block4(out)

        out = self.block5(out)

        out = self.conv5(out)
        out = self.norm5(out)
        out = ops.relu(out)
        out = self.block6(out)

        out = self.block7(out)

        out = ops.adaptive_avg_pool2d(out, 1)
        out = out.view(out.shape[0], -1)

        yaw = self.fc_roll(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_yaw(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
