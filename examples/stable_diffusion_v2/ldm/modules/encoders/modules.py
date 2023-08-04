# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os

import numpy as np
from ldm.models.clip.simple_tokenizer import get_tokenizer

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor

from .text_encoder import TextEncoder

SD_VERSION = os.getenv("SD_VERSION", default="2.0")


class FrozenCLIPEmbedder(nn.Cell):
    def __init__(self, max_length=77, use_fp16=False):
        super(FrozenCLIPEmbedder, self).__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.max_length = max_length

        self.tokenizer = get_tokenizer(SD_VERSION)  # TODO: reuse the instantiated tokenized used in load dataset

        if SD_VERSION.startswith("1."):
            self.transformer = TextEncoder(
                context_length=77, vocab_size=49408, output_dim=768, width=768, layers=12, heads=12, dtype=self.dtype
            )
        else:
            # TODO: in original implementation, layers=24. But it seems the weights in the last layer are not converted
            #   yet and leads to worse generalization result.
            self.transformer = TextEncoder(
                context_length=77, vocab_size=49408, output_dim=1024, width=1024, layers=23, heads=16, dtype=self.dtype
            )
            # self.transformer = TextEncoder(
            # context_length=77, vocab_size=49408, output_dim=1024, width=1024, layers=24, heads=16, dtype=self.dtype)

    def tokenize(self, texts):
        SOT_TEXT = self.tokenizer.sot_text
        EOT_TEXT = self.tokenizer.eot_text
        CONTEXT_LEN = 77

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
