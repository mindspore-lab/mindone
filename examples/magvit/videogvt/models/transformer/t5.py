import mindspore as ms
from mindspore import nn

from transformers import T5Tokenizer
from transformers.models.t5.configuration_t5 import T5Config

from mindone.mindone.transformers import T5EncoderModel

from beartype import beartype
from typing import List, Union


def exists(val):
    return val is not None


# config

MAX_LENGTH = 256

# DEFAULT_T5_NAME = "google/t5-v1_1-base"
DEFAULT_T5_NAME = "/disk3/katekong/text_encoders/models--DeepFloyd--t5-v1_1-xxl"

T5_CONFIGS = {}

# singleton globals


def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer


def get_model(name):
    config = T5Config.from_pretrained(name)
    model = T5EncoderModel(config)
    # model = T5EncoderModel.from_pretrained(name)
    return model


def get_model_and_tokenizer(name):
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()
    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]["model"], T5_CONFIGS[name]["tokenizer"]


def get_encoded_dim(name):
    if name not in T5_CONFIGS:
        # avoids loading the model if we only want to get the dim
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config
    else:
        assert False
    return config.d_model


# encoding text


@beartype
def t5_encode_text(
    texts: Union[str, List[str]],
    name=DEFAULT_T5_NAME,
):
    if isinstance(texts, str):
        texts = [texts]

    t5, tokenizer = get_model_and_tokenizer(name)
    t5.set_train(False)

    encoded = tokenizer.batch_encode_plus(
        texts, padding="longest", max_length=MAX_LENGTH, truncation=True
    )

    input_ids = ms.Tensor(encoded.input_ids)
    attn_mask = ms.Tensor(encoded.attention_mask)

    encoded_text = t5(input_ids=input_ids, attention_mask=attn_mask)

    attn_mask = attn_mask.bool()
    encoded_text = encoded_text.masked_fill(~attn_mask[..., None], 0.0)

    return encoded_text


class TextEncoder(nn.Cell):
    def __init__(self, model_name=DEFAULT_T5_NAME):
        super().__init__()
        t5, tokenizer = get_model_and_tokenizer(model_name)
        t5.set_train(False)

        self.encoder = t5
        self.tokenizer = tokenizer

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer.batch_encode_plus(
            texts, padding="longest", max_length=MAX_LENGTH, truncation=True
        )

        input_ids = ms.Tensor(encoded.input_ids)
        attn_mask = ms.Tensor(encoded.attention_mask)

        encoded_text = self.encoder(input_ids=input_ids, attention_mask=attn_mask)

        attn_mask = attn_mask.bool()
        encoded_text = encoded_text.masked_fill(~attn_mask[..., None], 0.0)

        return encoded_text
