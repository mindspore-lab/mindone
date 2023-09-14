import os
import json

import mindspore as ms

from mindnlp.transforms import T5Tokenizer
from mindnlp.models import T5EncoderModel
from mindnlp.models.t5 import T5Config


__all__ = [
    "T5Tokenizer",
    "get_t5_tokenizer",
    "get_t5_encoder",
]


def get_t5_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained(os.path.join("flan_t5_large", "tokenizer.json"))
    return tokenizer

def get_t5_encoder(trainable=False):
    with open(os.path.join("flan_t5_large", "config.json"), "r") as file:
        config = json.load(file)
        config = T5Config(**config)
    model = T5EncoderModel(config).get_encoder()
    if not trainable:
        model.set_train(False)
        for param in model.get_parameters():
            param.requires_grad = False

    return model
