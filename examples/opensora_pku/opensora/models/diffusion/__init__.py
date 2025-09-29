# Adapted from
# https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/diffusion/__init__.py

from .opensora.modeling_opensora import OpenSora_v1_3_models, OpenSora_v1_3_models_class

Diffusion_models = {}
Diffusion_models.update(OpenSora_v1_3_models)


Diffusion_models_class = {}
Diffusion_models_class.update(OpenSora_v1_3_models_class)
