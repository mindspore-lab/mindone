# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This code is adapted from https://github.com/3DTopia/OpenLRM/
# to run openlrm on mindspore.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# ******************************************************************************
#   Code modified by Zexin He in 2023-2024.
#   Modifications are marked with clearly visible comments
#   licensed under the Apache License, Version 2.0.
# ******************************************************************************

# ********************************************************
from .attention import MemEffAttention

# ********** Modified by Zexin He in 2023-2024 **********
# Avoid using nested tensor for now, deprecating usage of NestedTensorBlock
from .block import Block, BlockWithModulation

# from .dino_head import DINOHead # not use
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
