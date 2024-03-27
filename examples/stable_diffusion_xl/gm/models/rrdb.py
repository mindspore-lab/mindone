# Copyright © 2023 Huawei Technologies Co, Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from gm.modules.common.resblock_with_input_conv import make_layer

import mindspore as ms
import mindspore.nn as nn


class RRDBNet(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels,
        num_blocks,
        scale,
        internal_ch=32,
        bias=False,
        get_feat=False,
        scale_to_0_1=False,
    ):
        super(RRDBNet, self).__init__()

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, padding=1, pad_mode="pad", has_bias=bias)
        self.RRDB_trunk = make_layer(RRDB, num_blocks, nf=mid_channels, gc=internal_ch)
        self.RRDB_trunk_list = []
        for rrdb_block in self.RRDB_trunk.cell_list:
            self.RRDB_trunk_list.append(rrdb_block)
        self.trunk_conv = nn.Conv2d(mid_channels, mid_channels, 3, 1, padding=1, pad_mode="pad", has_bias=bias)

        self.scale = scale
        if not get_feat:
            if scale >= 2:
                self.upconv1 = ResidualDenseBlock5C(mid_channels, mid_channels // 2)
            if scale >= 4:
                self.upconv2 = ResidualDenseBlock5C(mid_channels, mid_channels // 2)
            if scale == 8:
                self.upconv3 = nn.Conv2d(mid_channels, mid_channels, 3, 1, padding=1, pad_mode="pad", has_bias=bias)

            self.HRconv = nn.Conv2d(mid_channels, mid_channels, 3, 1, padding=1, pad_mode="pad", has_bias=bias)
            self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, 1, padding=1, pad_mode="pad", has_bias=bias)
            self.lrelu = nn.LeakyReLU(alpha=0.2)

        self.get_feat = get_feat
        self.scale_to_0_1 = scale_to_0_1

    def construct(self, x):
        h0, w0 = x.shape[-2:]
        if self.scale_to_0_1:
            x = (x + 1) / 2
        feats = []
        feat = self.conv_first(x)
        feat_first = feat
        for rrdb_block in self.RRDB_trunk_list:
            feat = rrdb_block(feat)
            feats.append(feat)
        trunk = self.trunk_conv(feat)
        feat = feat_first + trunk
        if self.get_feat:
            feats.append(feat)
            return feats

        if self.scale >= 2:
            feat = ms.ops.interpolate(
                feat, size=(feat.shape[2] * 2, feat.shape[3] * 2), align_corners=False, mode="bilinear"
            )
            feat = self.upconv1(feat)
            feat = self.lrelu(feat)
        if self.scale >= 4:
            feat = ms.ops.interpolate(
                feat, size=(feat.shape[2] * 2, feat.shape[3] * 2), align_corners=False, mode="bilinear"
            )
            feat = self.upconv2(feat)
            feat = self.lrelu(feat)
        if self.scale == 8:
            feat = ms.ops.interpolate(
                feat, size=(feat.shape[2] * 2, feat.shape[3] * 2), align_corners=False, mode="bilinear"
            )
            feat = self.upconv3(feat)
            feat = self.lrelu(feat)
        feat_hr = self.HRconv(feat)
        out = self.lrelu(feat_hr)
        out = self.conv_last(out)
        out = out.clip(0, 1)
        if self.scale_to_0_1:
            out = out * 2 - 1
        return out


class ResidualDenseBlock5C(nn.Cell):
    def __init__(self, nf=64, gc=32, bias=False):
        super(ResidualDenseBlock5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, padding=1, pad_mode="pad", has_bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, padding=1, pad_mode="pad", has_bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, padding=1, pad_mode="pad", has_bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, padding=1, pad_mode="pad", has_bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, padding=1, pad_mode="pad", has_bias=bias)
        self.lrelu = nn.LeakyReLU(alpha=0.2)

    def construct(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(ms.ops.concat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(ms.ops.concat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(ms.ops.concat((x, x1, x2, x3), 1)))
        x5 = self.conv5(ms.ops.concat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Cell):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock5C(nf, gc)
        self.RDB2 = ResidualDenseBlock5C(nf, gc)
        self.RDB3 = ResidualDenseBlock5C(nf, gc)

    def construct(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
