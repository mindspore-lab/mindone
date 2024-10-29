# Adapted from examples/stable_diffusion_xl/gm/ & https://github.com/Stability-AI/generative-models/sgm
# Adatped from examples/stable_diffusion_xl
from .modules.embedders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "sgm.modules.embedders.GeneralConditioner",
    "params": {"emb_models": []},
}
