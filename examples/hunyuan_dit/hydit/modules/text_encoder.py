from transformers import AutoTokenizer

import mindspore as ms
from mindspore import nn, ops

from mindone.transformers import T5EncoderModel, T5ForConditionalGeneration


class MT5Embedder(nn.Cell):
    available_models = ["t5-v1_1-xxl"]

    def __init__(
        self,
        model_dir="t5-v1_1-xxl",
        model_kwargs=None,
        mindspore_dtype=None,
        use_tokenizer_only=False,
        conditional_generation=False,
        max_length=128,
    ):
        super().__init__()
        self.mindspore_dtype = mindspore_dtype or ms.bfloat16
        self.max_length = max_length
        if model_kwargs is None:
            model_kwargs = {
                "mindspore_dtype": self.mindspore_dtype,
            }
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if use_tokenizer_only:
            return
        if conditional_generation:
            self.model = None
            self.generation_model = T5ForConditionalGeneration.from_pretrained(model_dir)
            return
        self.model = T5EncoderModel.from_pretrained(model_dir, **model_kwargs).to(self.mindspore_dtype)

    def get_tokens_and_mask(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="np",
        )
        tokens = ms.tensor(text_tokens_and_mask["input_ids"])[0]
        mask = ms.tensor(text_tokens_and_mask["attention_mask"])[0]
        return tokens, mask

    def get_text_embeddings(self, texts, attention_mask=True, layer_index=-1):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="np",
        )

        outputs = self.model(
            input_ids=ms.tensor(text_tokens_and_mask["input_ids"]),
            attention_mask=ms.tensor(text_tokens_and_mask["attention_mask"]) if attention_mask else None,
            output_hidden_states=True,
        )
        text_encoder_embs = ops.stop_gradient(outputs[1][layer_index])

        return text_encoder_embs, ms.tensor(text_tokens_and_mask["attention_mask"])

    def __call__(self, tokens, attention_mask, layer_index=-1):
        outputs = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        z = ops.stop_gradient(outputs[1][layer_index])
        return z

    def general(self, text: str):
        input_ids = ms.tensor(self.tokenizer(text, max_length=128, return_tensors="np").input_ids)
        print(input_ids)
        outputs = self.generation_model(input_ids)
        return outputs
