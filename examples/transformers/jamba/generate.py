from time import time

from transformers import AutoModelForCausalLM, AutoTokenizer

import mindspore as ms
from mindspore import Tensor


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "ai21labs/AI21-Jamba-Large-1.6",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "ai21labs/AI21-Jamba-Large-1.6", mindspore_dtype=ms.float16, device_map="auto", attn_implementation="sdpa"
    )
    input_ids = tokenizer("Plants create energy through a process known as", return_tensors="np")
    input_ids = (
        Tensor(input_ids) if (len(input_ids.shape) == 2 and input_ids.shape[0] == 1) else Tensor(input_ids).unsqueeze(0)
    )  # (1, L)
    infer_start = time()
    output = model.generate(**input_ids, cache_implementation="static")
    print(f"Inference time: {time() - infer_start:.3f}s")
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
