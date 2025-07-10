from transformers import HerbertTokenizer

from mindspore import Tensor

from mindone.transformers import RobertaModel

model_names = {
    "herbert-klej-cased-v1": {
        "tokenizer": "allegro/herbert-klej-cased-tokenizer-v1",
        "model": "allegro/herbert-klej-cased-v1",
    },
    "herbert-base-cased": {
        "tokenizer": "allegro/herbert-base-cased",
        "model": "allegro/herbert-base-cased",
    },
    "herbert-large-cased": {
        "tokenizer": "allegro/herbert-large-cased",
        "model": "allegro/herbert-large-cased",
    },
}

tokenizer = HerbertTokenizer.from_pretrained(model_names["herbert-klej-cased-v1"]["tokenizer"])
model = RobertaModel.from_pretrained(model_names["herbert-klej-cased-v1"]["model"])

encoded_input = tokenizer.encode("Kto ma lepszą sztukę, ma lepszy rząd – to jasne.", return_tensors="np")

encoded_input = (
    Tensor(encoded_input)
    if (len(encoded_input.shape) == 2 and encoded_input.shape[0] == 1)
    else Tensor(encoded_input).unsqueeze(0)
)  # (1, L)
outputs = model(encoded_input)
print(outputs)
