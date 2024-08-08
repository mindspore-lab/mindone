# -*- coding: utf-8 -*-
import html
import logging
import re
import urllib.parse as ul
from typing import Optional, Sequence, Tuple, Union

import ftfy
import numpy as np
from bs4 import BeautifulSoup

import mindspore as ms
from mindspore import Tensor, nn

from .t5_modeling import get_t5_encoder, get_t5_tokenizer

logger = logging.getLogger(__name__)


__all__ = ["T5Tokenizer", "T5Encoder", "T5Embedder"]


class T5Tokenizer:
    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + r"\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    def __init__(self, cache_dir: str, use_text_preprocessing: bool = True, max_length: int = 120):
        self.use_text_preprocessing = use_text_preprocessing
        self.cache_dir = cache_dir
        self.tokenizer = get_t5_tokenizer(cache_dir)
        self.max_length = max_length
        self.tokenizer.context_length = max_length

    def __call__(
        self, texts: Union[str, Sequence[str]], return_tensor: bool = True
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray]]:
        if isinstance(texts, str):
            texts = [texts]
        texts = [self._text_preprocessing(text) for text in texts]
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
        )
        text_tokens = text_tokens_and_mask["input_ids"]
        mask = text_tokens_and_mask["attention_mask"]
        if return_tensor:
            return Tensor(text_tokens, dtype=ms.int32), Tensor(mask, dtype=ms.float32)
        else:
            return text_tokens, mask

    def _text_preprocessing(self, text: str) -> str:
        if self.use_text_preprocessing:
            # The exact text cleaning as was in the training stage:
            text = self.clean_caption(text)
            text = self.clean_caption(text)
            return text
        else:
            return text.lower().strip()

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()


class T5Encoder(nn.Cell):
    def __init__(self, cache_dir: str, pretrained_ckpt: Optional[str] = None) -> None:
        super().__init__()
        model = get_t5_encoder(cache_dir)
        if pretrained_ckpt:
            # load t5 ckpt into self.model
            logger.info("Loading t5 checkpoint from {}".format(pretrained_ckpt))
            param_dict = ms.load_checkpoint(pretrained_ckpt)
            param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict)
            assert len(param_not_load) == 0 and len(ckpt_not_load) == 1  # shared.embedding_table
        self.model = model

    def construct(self, text_tokens: Tensor, mask: Tensor = None) -> Tensor:
        text_encoder_embs = self.model(
            input_ids=text_tokens,
            attention_mask=mask,
        )
        if isinstance(text_encoder_embs, (list, tuple)):
            text_encoder_embs = text_encoder_embs[0]
        return text_encoder_embs


class T5Embedder(nn.Cell):
    def __init__(
        self,
        cache_dir: str,
        use_text_preprocessing: bool = True,
        model_max_length: int = 120,
        pretrained_ckpt: Optional[str] = None,
    ) -> None:
        super().__init__(auto_prefix=False)
        self.tokenizer = T5Tokenizer(
            cache_dir, use_text_preprocessing=use_text_preprocessing, max_length=model_max_length
        )
        self.encoder = T5Encoder(cache_dir, pretrained_ckpt=pretrained_ckpt)

    def get_text_embeddings(self, texts: Union[str, Sequence[str]]) -> Tuple[Tensor, Tensor]:
        text_tokens, mask = self.tokenizer(texts)
        text_encoder_embs = self.encoder(text_tokens, mask)
        return text_encoder_embs, mask

    def construct(self, text_tokens: Tensor, mask: Tensor = None) -> Tensor:
        text_encoder_embs = self.encoder(text_tokens, mask)
        return text_encoder_embs
