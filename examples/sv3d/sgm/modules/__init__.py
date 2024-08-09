from .embedders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "gm.modules.embedders.GeneralConditioner",
    "params": {"emb_models": []},
}
