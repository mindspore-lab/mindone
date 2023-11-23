"""This script defines deep neural networks for Deep3DFaceRecon_pytorch
"""

import os
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Initializer, Constant
from models.face3d.facexlib.resnet import conv1x1, resnet50

from typing import Type, Any, Callable, Union, List, Optional
# from kornia.geometry import warp_affine


def resize_n_crop(image, M, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    return warp_affine(image, M, dsize=(dsize, dsize), align_corners=True)


def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict


def define_net_recon(net_recon, use_last_fc=False, init_path=None):
    return ReconNetWrapper(net_recon, use_last_fc=use_last_fc, init_path=init_path)


class ReconNetWrapper(nn.Cell):
    fc_dim = 257

    def __init__(self, net_recon, use_last_fc=False, init_path=None):
        super(ReconNetWrapper, self).__init__()
        self.use_last_fc = use_last_fc
        if net_recon not in func_dict:
            return NotImplementedError('network [%s] is not implemented', net_recon)
        func, last_dim = func_dict[net_recon]
        backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)
        if init_path and os.path.isfile(init_path):
            state_dict = filter_state_dict(
                ms.load_state_dict(init_path))
            ms.load_param_into_net(backbone, state_dict)
            print("loading init net_recon %s from %s" % (net_recon, init_path))
        self.backbone = backbone
        if not use_last_fc:
            self.final_layers = nn.CellList([
                conv1x1(last_dim, 80, bias=True),  # id layer
                conv1x1(last_dim, 64, bias=True),  # exp layer
                conv1x1(last_dim, 80, bias=True),  # tex layer
                conv1x1(last_dim, 3, bias=True),  # angle layer
                conv1x1(last_dim, 27, bias=True),  # gamma layer
                conv1x1(last_dim, 2, bias=True),  # tx, ty
                conv1x1(last_dim, 1, bias=True)   # tz
            ])
            for m in self.final_layers:
                m.weight_init = Initializer(init=Constant(0))
                m.bias_init = Initializer(init=Constant(0))

    def construct(self, x):
        x = self.backbone(x)
        if not self.use_last_fc:
            output = []
            for layer in self.final_layers:
                output.append(layer(x))
            x = ops.flatten(ops.cat(output, axis=1), start_dim=1)
        return x


func_dict = {
    # 'resnet18': (resnet18, 512),
    'resnet50': (resnet50, 2048)
}
