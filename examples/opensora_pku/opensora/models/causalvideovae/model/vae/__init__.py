# Adapted from
# https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/model/vae/__init__.py

import logging

from .modeling_causalvae import CausalVAEModel
from .modeling_wfvae import WFVAEModel

logger = logging.getLogger(__name__)
