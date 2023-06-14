import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from collections import namedtuple
import mindcv


class LPIPS(nn.Cell):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        # self.net = nn.Identity()
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        self.set_train(False)
        for param in self.trainable_params():
            param.requires_grad = False
        self.cast = ops.Cast()

    def load_from_pretrained(self, ckpt_path="vgg_lpips"):
        print("loaded pretrained LPIPS loss from {}".format(ckpt_path))

    def construct(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        val = ms.Tensor(0, dtype=input.dtype)
        for kk in range(len(self.chns)):
            val += spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
        return val
        # return self.cast(val, input.dtype)


class ScalingLayer(nn.Cell):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.shift = ms.Tensor([-.030, -.088, -.188])[None, :, None, None]
        self.scale = ms.Tensor([.458, .448, .450])[None, :, None, None]

    def construct(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Cell):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, has_bias=False), ]
        self.model = nn.SequentialCell(layers)


class vgg16(nn.Cell):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        model = mindcv.create_model('vgg16', pretrained=pretrained)
        model.set_train(False)
        vgg_pretrained_features = model.features
        # vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.SequentialCell()
        self.slice2 = nn.SequentialCell()
        self.slice3 = nn.SequentialCell()
        self.slice4 = nn.SequentialCell()
        self.slice5 = nn.SequentialCell()
        self.N_slices = 5
        for x in range(4):
            self.slice1.append(vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.append(vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.append(vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.append(vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.append(vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.trainable_params():
                param.requires_grad = False

    def construct(self, X):
        h = self.slice1(X)
        h_relu1_2 = ops.stop_gradient(h)
        h = self.slice2(h)
        h_relu2_2 = ops.stop_gradient(h)
        h = self.slice3(h)
        h_relu3_3 = ops.stop_gradient(h)
        h = self.slice4(h)
        h_relu4_3 = ops.stop_gradient(h)
        h = self.slice5(h)
        h_relu5_3 = ops.stop_gradient(h)
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = ops.sqrt(ops.reduce_sum(x**2, axis=1)[:, None])
    return x / (norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keep_dims=keepdim)
