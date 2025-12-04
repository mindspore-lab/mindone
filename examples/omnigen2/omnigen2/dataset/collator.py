from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray


class OmniGen2Collator:
    def __init__(self, tokenizer, max_token_len: int):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __call__(self, instructions: "ndarray") -> tuple["ndarray", "ndarray"]:
        text_inputs = self.tokenizer(
            instructions.tolist(),
            padding="longest",
            max_length=self.max_token_len,
            truncation=True,
            return_tensors="np",
        )
        return text_inputs.input_ids, text_inputs.attention_mask
