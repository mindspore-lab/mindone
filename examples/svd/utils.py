import random
import sys

import numpy as np
from modules.encoders.modules import VideoPredictionEmbedderWithEncoder

import mindspore as ms
from mindspore import amp
from mindspore.nn import GroupNorm, SiLU, Softmax

sys.path.append("../stable_diffusion_xl")  # FIXME: loading modules from the SDXL directory
from gm.modules.diffusionmodules.util import GroupNorm as GroupNorm3D


def _mixed_precision(network):
    black_list = amp.get_black_list() + [SiLU, GroupNorm, GroupNorm3D, Softmax]
    return amp.custom_mixed_precision(network, black_list=black_list)


def mixed_precision(net):
    cells = net.name_cells()
    for cell in cells:
        # don't set the loss_fn and conditioner to mixed precision (each embedder has its own amp level)
        if (
            not cells[cell] is net.loss_fn
            and not cells[cell] is net.conditioner
            and not (net.disable_first_stage_amp and cells[cell] is net.first_stage_model)
        ):
            setattr(net, cell, _mixed_precision(cells[cell]))
    for emb in net.conditioner.embedders:
        if not (isinstance(emb, VideoPredictionEmbedderWithEncoder) and emb.disable_encoder_amp):
            _mixed_precision(emb)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
