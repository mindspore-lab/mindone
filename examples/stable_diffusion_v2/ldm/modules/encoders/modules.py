import numpy as np
from ldm.models.clip.simple_tokenizer import get_tokenizer

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor

from .text_encoder import TextEncoder


class FrozenCLIPEmbedder(nn.Cell):
    def __init__(
        self,
        use_fp16=False,
        tokenizer_name="WordpieceTokenizer",
        context_length=77,
        vocab_size=49408,
        output_dim=768,
        width=768,
        layers=12,
        heads=12,
        epsilon=1e-5,
        use_quick_gelu=False,
    ):
        super(FrozenCLIPEmbedder, self).__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.context_length = context_length
        self.tokenizer = get_tokenizer(tokenizer_name)
        setattr(self.tokenizer, "context_length", context_length)

        self.transformer = TextEncoder(
            context_length=context_length,
            vocab_size=vocab_size,
            output_dim=output_dim,
            width=width,
            layers=layers,
            heads=heads,
            epsilon=epsilon,
            use_quick_gelu=use_quick_gelu,
            dtype=self.dtype,
        )

    def tokenize(self, texts):
        SOT_TEXT = self.tokenizer.sot_text
        EOT_TEXT = self.tokenizer.eot_text
        CONTEXT_LEN = self.context_length

        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = np.zeros((len(all_tokens), CONTEXT_LEN), np.int64)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > CONTEXT_LEN:
                tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]

            result[i, : len(tokens)] = np.array(tokens, np.int64)

        return Tensor(result)

    def encode(self, tokenized_text):
        outputs = self.transformer(tokenized_text)
        return outputs

    def construct(self, c):
        outputs = self.transformer(c)
        return outputs
