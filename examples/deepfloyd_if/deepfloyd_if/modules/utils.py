# -*- coding: utf-8 -*-
import numpy as np
from mindspore import nn
from mindspore.dataset import transforms, vision


def predict_proba(X, weights, biases):
    logits = X @ weights.T + biases
    proba = np.where(logits >= 0, 1 / (1 + np.exp(-logits)), np.exp(logits) / (1 + np.exp(logits)))
    return proba.T


def load_model_weights(path):
    model_weights = np.load(path)
    return model_weights['weights'], model_weights['biases']


def clip_process_generations(generations):
    min_size = min(generations.shape[-2:])
    return transforms.Compose([
        vision.CenterCrop(min_size),
        vision.Resize(224, interpolation=vision.Inter.BICUBIC),  # , antialias=True
        vision.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])(generations)


def get_pt2ms_mappings(model: nn.Cell):
    # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)
    mappings = {}

    def check_key(k):
        if k in mappings:
            raise KeyError(f"param name {k} is already in mapping!")

    for name, cell in model.cells_and_names():
        if isinstance(cell, nn.Conv1d):
            check_key(f"{name}.weight")
            mappings[f"{name}.weight"] = f"{name}.weight", lambda x: np.expand_dims(x, axis=-2)
        elif isinstance(cell, nn.Embedding):
            check_key(f"{name}.weight")
            mappings[f"{name}.weight"] = f"{name}.embedding_table", lambda x: x
        elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            check_key(f"{name}.weight")
            mappings[f"{name}.weight"] = f"{name}.gamma", lambda x: x
            check_key(f"{name}.bias")
            mappings[f"{name}.bias"] = f"{name}.beta", lambda x: x
    return mappings
