import logging
import os

import mindcv
from mg.utils.load_models import load_from_pretrained

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

_logger = logging.getLogger(__name__)


class LPIPS(nn.Cell):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vgg16 features
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        # load NetLin metric layers
        self.load_lpips()

        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.lins = nn.CellList(self.lins)

        # create vision backbone and load pretrained weights
        self.net = vgg16(pretrained=True, requires_grad=False)

        self.set_train(False)
        for param in self.trainable_params():
            param.requires_grad = False

    def load_lpips(self, ckpt_path="models/lpips_vgg-426bf45c.ckpt"):
        if not os.path.exists(ckpt_path):
            ckpt_path = "https://download-mindspore.osinfra.cn/toolkits/mindone/autoencoders/lpips_vgg-426bf45c.ckpt"
            load_from_pretrained(self, ckpt_path)

        _logger.info("loaded pretrained LPIPS loss from {}".format(ckpt_path))

    def construct(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        val = 0  # ms.Tensor(0, dtype=input.dtype)
        for kk in range(len(self.chns)):
            diff = (normalize_tensor(outs0[kk]) - normalize_tensor(outs1[kk])) ** 2
            # res += spatial_average(lins[kk](diff), keepdim=True)
            # lin_layer = lins[kk]
            val += ops.mean(self.lins[kk](diff), axis=[2, 3], keep_dims=True)
        return val


class ScalingLayer(nn.Cell):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.shift = ms.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        self.scale = ms.Tensor([0.458, 0.448, 0.450])[None, :, None, None]

    def construct(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Cell):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False, dtype=ms.float32):
        super(NetLinLayer, self).__init__()
        # TODO: can parse dtype=dtype in ms2.3
        layers = (
            [
                nn.Dropout(p=0.5).to_float(dtype),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, has_bias=False).to_float(dtype),
        ]
        self.model = nn.SequentialCell(layers)

    def construct(self, x):
        return self.model(x)


class vgg16(nn.Cell):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        # FIXME: add bias in vgg. use the same model weights in PT.
        model = mindcv.create_model("vgg16", pretrained=pretrained)
        model.set_train(False)
        vgg_pretrained_features = model.features
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
            for param in model.trainable_params():
                param.requires_grad = False

    def construct(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        out = (h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = ops.sqrt((x**2).sum(1, keepdims=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keep_dims=keepdim)
