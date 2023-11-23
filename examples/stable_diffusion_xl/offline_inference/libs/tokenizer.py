from typing import List, Union

import numpy as np
from gm.modules.embedders.open_clip import tokenize as openclip_tokenize
from transformers import CLIPTokenizer


class OpenCLIPTokenizer:
    def __call__(self, texts: Union[str, List[str]], context_length: int = 77) -> np.ndarray:
        return openclip_tokenize(texts, context_length)[0]


class IdentityTokenizer:
    def __call__(self, x):
        return x


def get_tokenizer(tokenizer_name, version="openai/clip-vit-large-patch14"):
    if tokenizer_name == "CLIPTokenizer":
        tokenizer = CLIPTokenizer.from_pretrained(version)
    elif tokenizer_name == "OpenCLIPTokenizer":
        tokenizer = OpenCLIPTokenizer()
    elif tokenizer_name == "IdentityTokenizer":
        tokenizer = IdentityTokenizer()
    else:
        raise NotImplementedError(f"tokenizer {tokenizer_name} not implemented")
    return tokenizer
