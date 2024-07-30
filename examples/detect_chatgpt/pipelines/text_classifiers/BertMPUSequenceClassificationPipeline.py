import json
import os

import numpy as np

import mindspore as ms

from .preprocess import en_cleaning, zh_cleaning


class BertMPUSequenceClassificationPipeline:
    def __init__(
        self,
        model_name="bert_base",
        config_path="config.json",
        tokenizer_path="tokenizer.json",
        amp_level="O1",
    ):
        super().__init__()
        with open(config_path, "r") as file:
            config = json.load(file)
            self.max_sequence_length = config.get("max_sequence_length", 512)

        if model_name == "bert_base":
            from mindnlp._legacy.amp import auto_mixed_precision
            from mindnlp.models import BertForSequenceClassification
            from mindnlp.models.bert import BertConfig
            from mindnlp.transforms import BertTokenizer

            pad_token_id = config.get("pad_token_id", 0)
            self.config = BertConfig(**config)
            self.tokenizer = BertTokenizer(tokenizer_path)
            self.backbone = BertForSequenceClassification(self.config)
            self.backbone = auto_mixed_precision(self.backbone, amp_level)
            self.clean = zh_cleaning
        elif model_name == "roberta_base":
            from mindnlp._legacy.amp import auto_mixed_precision
            from mindnlp.models import RobertaForSequenceClassification
            from mindnlp.models.roberta import RobertaConfig
            from mindnlp.transforms import RobertaTokenizer

            pad_token_id = config.get("pad_token_id", 1)
            self.config = RobertaConfig(**config)
            self.tokenizer = RobertaTokenizer(tokenizer_path)
            self.backbone = RobertaForSequenceClassification(self.config)
            self.backbone = auto_mixed_precision(self.backbone, amp_level)
            self.clean = en_cleaning
        self.tokenizer._pad_token = pad_token_id

        self.rng = np.random.RandomState(0)

    def load_from_pretrained(self, ckpt_path: str):
        if os.path.exists(ckpt_path):
            ms.load_checkpoint(ckpt_path, self.backbone, strict_load=False)

    def tokenize_truncate_pad(self, text: str):
        tokens = self.tokenizer(text).tolist()
        output_length = min(len(tokens), self.max_sequence_length)
        start = 0 if len(tokens) <= output_length else self.rng(0, len(tokens) - output_length + 1)
        end = start + output_length
        tokens = tokens[start:end]

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = ms.Tensor(tokens + padding)
        mask = ms.ops.ones(tokens.shape[0], ms.int32)
        mask[-len(padding) :] = 0

        return tokens[None, ...], mask[None, ...]

    def preprocess(self, text: str):
        text = self.clean(text)
        x, mask = self.tokenize_truncate_pad(text)
        return x, mask

    def predict(self, text: str):
        x, mask = self.preprocess(text)
        logits = self.backbone(x, attention_mask=mask, labels=None)
        logits = logits[0]  # remove tuple
        logits = ms.ops.softmax(logits, axis=-1)[0]  # remove batch
        return {
            "human": logits[0].asnumpy(),
            "machine": logits[1].asnumpy(),
        }
