

from mindnlp.models.roberta import RobertaConfig
from mindnlp.transforms import RobertaTokenizer
from mindnlp.models import RobertaForSequenceClassification

from mindone.preprocess import en_cleaning
import json
import numpy as np
import mindspore as ms


class RobertaSequenceClassificationPipeline():
    def __init__(
        self,
        model_name='roberta_base',
        config_path='config.json',
        tokenizer_path='tokenizer.json',
    ):
        super().__init__()
        if model_name != 'roberta_base':
            raise NotImplementedError

        with open(config_path, 'r') as file:
            config = json.load(file)
            self.max_sequence_length = config.get('max_sequence_length', 512)
            ckpt_path = config.get('ckpt_path', 'roberta_18plus.ckpt')
            pad_token_id = config.get('pad_token_id', 1)
            self.config = RobertaConfig(**config)

        self.tokenizer = RobertaTokenizer(tokenizer_path)
        self.tokenizer._pad_token = pad_token_id

        self.roberta = RobertaForSequenceClassification(self.config)
        ms.load_checkpoint(ckpt_path, self.roberta, strict_load=True)

        self.rng = np.random.RandomState(0)
        self.label_to_meaning = ['human written', 'machine generated']

    def tokenize_truncate_pad(self, text: str):
        tokens = self.tokenizer.encode(text).ids
        output_length = min(len(tokens), self.max_sequence_length)
        start = 0 if len(tokens) <= output_length else self.rng(0, len(tokens) - output_length + 1)
        end = start + output_length
        tokens = tokens[start: end]

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = ms.Tensor(tokens + padding)
        mask = ms.ops.ones(tokens.shape[0], ms.int32)
        mask[-len(padding):] = 0

        return tokens[None, ...], mask[None, ...]

    def preprocess(self, text: str):
        text = en_cleaning(text)
        x, mask = self.tokenize_truncate_pad(text)
        return x, mask

    def predict(self, text: str):
        x, mask = self.preprocess(text)
        logits = self.roberta(x, attention_mask=mask, labels=None)
        logits = logits[0] # remove tuple
        logits = ms.ops.softmax(logits, axis=-1)[0] # remove batch
        return {
            'human': logits[0].asnumpy(),
            'machine': logits[1].asnumpy(),
        }
