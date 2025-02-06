# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
from typing import Optional

import mindspore as ms
from mindspore import mint, nn

from mindone.transformers import MSPreTrainedModel

from .dinov2.hub.backbones import dinov2_vitb14

logger = logging.getLogger("dinov2")


class FrozenDinoV2ImageEmbedder(nn.Cell):
    """
    Uses the dinov2 image encoder with camera modulation.
    Not actually frozen... If you want that set cond_stage_trainable=False in cfg
    """

    def __init__(
        self,
        version="dinov2_vitb14",
        ckpt_path=None,
        lrm_mode="plain_lrm",
    ):
        super().__init__()
        self.lrm_mode = lrm_mode
        assert version in ["dinov2_vitb14", "dinov2_vits14", "dinov2_vitl14", "dinov2_vitg14"]

        self.model = dinov2_vitb14(pretrained=False)

        if ckpt_path is not None:
            self.load_pretrained(ckpt_path)
        else:
            print("None pretrained model for dinov2 encoder ...")

    def to(self, dtype: Optional[ms.Type] = None):
        for p in self.get_parameters():
            p.set_dtype(dtype)
        return self

    def load_pretrained(self, ckpt_path):
        print("Loading dinov2 encoder ...")

        self.model, loading_info = MSPreTrainedModel.from_pretrained(
            self.model, ckpt_path, output_loading_info=True, mindspore_dtype=ms.float16
        )
        print(loading_info)
        logger.info(loading_info)

    def construct(self, x, *args, **kwargs):
        ret = self.model.forward_features_with_camera(x, *args, **kwargs)
        output = mint.cat([ret["x_norm_clstoken"].unsqueeze(1), ret["x_norm_patchtokens"]], dim=1)
        return output
