from typing import Literal

from transformers import CLIPTokenizer, T5Tokenizer

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import tensor

from mindone.transformers import CLIPTextModel, T5EncoderModel


class HFEmbedder:
    def __init__(self, from_pretrained: str, max_length: int, dtype: Literal["fp32", "fp16", "bf16"] = "fp32"):
        dtype = {"fp32": mstype.float32, "fp16": mstype.float16, "bf16": mstype.bfloat16}[dtype]
        self.max_length = max_length

        if "openai" in from_pretrained:
            self.tokenizer = CLIPTokenizer.from_pretrained(from_pretrained, max_length=max_length)
            self.hf_module = CLIPTextModel.from_pretrained(from_pretrained, mindspore_dtype=dtype).set_train(False)
            self.output_id = 1
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(from_pretrained, max_length=max_length, legacy=True)
            self.hf_module = T5EncoderModel.from_pretrained(from_pretrained, mindspore_dtype=dtype).set_train(False)
            self.output_id = 0

    def __call__(self, text: list[str]) -> Tensor:
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
            input_ids=tensor(batch_encoding.input_ids, dtype=mstype.int32),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_id]
