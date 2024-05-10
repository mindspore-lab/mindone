# -*- coding: utf-8 -*-
import glob
import html
import json
import logging
import os
import re
import urllib.parse as ul

import ftfy
from bs4 import BeautifulSoup
from transformers import CLIPTokenizer
from transformers.models.clip.configuration_clip import CLIPTextConfig

import mindspore as ms

from mindone.transformers import CLIPTextModel

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """
    A class for embedding texts and images using a pretrained CLIP model.
    """

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        cache_dir="./cache_dir",
        use_text_preprocessing=True,
        max_length=77,
    ):
        """
        Initializes the CLIPEmbedder with specified model and configurations.
        """
        self.cache_dir = os.path.join(cache_dir, model_name)
        self.use_text_preprocessing = use_text_preprocessing
        self.max_length = max_length

        assert os.path.exists(self.cache_dir), f"Cache directory {self.cache_dir} does not exist."

        self.tokenizer = CLIPTokenizer.from_pretrained(self.cache_dir)
        with open(os.path.join(self.cache_dir, "config.json"), "r") as file:
            config = json.load(file)
            config = CLIPTextConfig(**config)
        text_model = CLIPTextModel(config)

        ckpt_path = glob.glob(os.path.join(self.cache_dir, "*.ckpt"))
        if len(ckpt_path) == 0:
            logger.info("No checkpoint found in the cache directory. Use random initialization.")
        else:
            assert len(ckpt_path) == 1, "Multiple checkpoints found in the cache directory."
            ckpt_path = ckpt_path[0]
            logger.info(f"Load checkpoint from {ckpt_path}.")
            param_dict = ms.load_checkpoint(ckpt_path)
            param_not_load, ckpt_not_load = ms.load_param_into_net(text_model, param_dict)
            if len(param_not_load) > 0:
                logger.warning(f"Parameter not loaded: {param_not_load}")
            if len(ckpt_not_load) > 0:
                logger.warning(f"Checkpoint not loaded: {ckpt_not_load}")

        self.text_model = text_model
        self.text_model.set_train(False)
        for param in self.text_model.get_parameters():
            param.requires_grad = False

    def get_text_embeddings(self, texts):
        """
        Generates embeddings for a list of text prompts.
        """
        self._validate_input_list(texts, str)

        if self.use_text_preprocessing:
            texts = [self._clean_text(text) for text in texts]

        embeddings, _ = self.encode_text(texts)

        return embeddings

    def encode_text(self, texts):
        """
        Encodes texts into embeddings and returns the last hidden state and pooled output.
        """
        self._validate_input_list(texts, str)

        batch_encoding = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding="max_length")
        text_input_ids = ms.Tensor(batch_encoding.input_ids)
        attention_mask = ms.Tensor(batch_encoding.attention_mask)
        text_features, _ = self.get_text_features(text_input_ids, attention_mask)
        return text_features

    def get_text_features(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs.last_hidden_state, outputs.pooler_output

    def _validate_input_list(self, input_list, expected_type):
        """
        Validates that the input is a list of expected type.
        """
        if not isinstance(input_list, list) or not all(isinstance(item, expected_type) for item in input_list):
            raise ValueError(f"Input must be a list of {expected_type.__name__}.")

    def _clean_text(self, text):
        """
        Applies basic cleaning and formatting to a text string.
        """
        text = ftfy.fix_text(text)
        text = html.unescape(text)
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

        caption = BeautifulSoup(caption, features="html.parser").text

        caption = re.sub(r"@[\w\d]+\b", "", caption)

        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)

        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        caption = re.sub(r"&quot;?", "", caption)

        caption = re.sub(r"&amp", "", caption)

        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        caption = re.sub(r"\\n", " ", caption)

        caption = re.sub(r"#\d{1,3}\b", "", caption)

        caption = re.sub(r"#\d{5,}\b", "", caption)
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)
        caption = re.sub(r"[\"\']{2,}", r'"', caption)
        caption = re.sub(r"[\.]{2,}", r" ", caption)

        caption = re.sub(self.bad_punct_regex, r" ", caption)
        caption = re.sub(r"\s+\.\s+", r" ", caption)
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

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()


if __name__ == "__main__":
    clip_embedder = CLIPEmbedder()

    # Example
    text_prompts = [
        "A photo of a cute puppy playing with a ball.",
        "An image of a beautiful sunset over the ocean.",
        "A scene depicting a busy city street.",
    ]
    text_embeddings = clip_embedder.get_text_embeddings(text_prompts)
    print(f"Text embeddings shape: {text_embeddings.shape}")
