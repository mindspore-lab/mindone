import mindspore.mint.nn.functional as F
from mindspore import mint, nn

from .resnet import ResNet18


class ConvBNReLU(nn.Cell):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = mint.nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = mint.nn.BatchNorm2d(out_chan)

    def construct(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x


class BiSeNetOutput(nn.Cell):
    def __init__(self, in_chan, mid_chan, num_class):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = mint.nn.Conv2d(mid_chan, num_class, kernel_size=1, bias=False)

    def construct(self, x):
        feat = self.conv(x)
        out = self.conv_out(feat)
        return out, feat


class AttentionRefinementModule(nn.Cell):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = mint.nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = mint.nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = mint.nn.Sigmoid()

    def construct(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.shape[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = mint.mul(feat, atten)
        return out


class ContextPath(nn.Cell):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet = ResNet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def construct(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        h8, w8 = feat8.shape[2:]
        h16, w16 = feat16.shape[2:]
        h32, w32 = feat32.shape[2:]

        avg = F.avg_pool2d(feat32, feat32.shape[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (h32, w32), mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (h16, w16), mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (h8, w8), mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16


class FeatureFusionModule(nn.Cell):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = mint.nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = mint.nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = mint.nn.ReLU()
        self.sigmoid = mint.nn.Sigmoid()

    def construct(self, fsp, fcp):
        fcat = mint.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.shape[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = mint.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class BiSeNet(nn.Cell):
    def __init__(self, num_class):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, num_class)
        self.conv_out16 = BiSeNetOutput(128, 64, num_class)
        self.conv_out32 = BiSeNetOutput(128, 64, num_class)

    def construct(self, x, return_feat=False):
        h, w = x.shape[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # return res3b1 feature
        feat_sp = feat_res8  # replace spatial path feature with res3b1 feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        out, feat = self.conv_out(feat_fuse)
        out16, feat16 = self.conv_out16(feat_cp8)
        out32, feat32 = self.conv_out32(feat_cp16)

        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
        out16 = F.interpolate(out16, (h, w), mode="bilinear", align_corners=True)
        out32 = F.interpolate(out32, (h, w), mode="bilinear", align_corners=True)

        if return_feat:
            feat = F.interpolate(feat, (h, w), mode="bilinear", align_corners=True)
            feat16 = F.interpolate(feat16, (h, w), mode="bilinear", align_corners=True)
            feat32 = F.interpolate(feat32, (h, w), mode="bilinear", align_corners=True)
            return out, out16, out32, feat, feat16, feat32
        else:
            return out, out16, out32
