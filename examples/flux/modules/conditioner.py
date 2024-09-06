from typing import List

from transformers import CLIPTokenizer, T5Tokenizer

from mindspore import Tensor

from mindone.transformers import CLIPTextModel, T5EncoderModel


class HFEmbedder(object):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        # self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        self.output_index = 1 if self.is_clip else 0

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module.set_train(False)

    def __call__(self, text: List[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="np",
        )

        outputs = self.hf_module(
            input_ids=Tensor.from_numpy(batch_encoding["input_ids"]),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_index]
