from time import time

from transformers import AutoTokenizer

import mindspore as ms
from mindspore import Tensor

from mindone.transformers import HeliumForCausalLM


def main():
    model_id = "kyutai/helium-1-preview-2b"  # "google/helium-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = HeliumForCausalLM.from_pretrained(model_id, mindspore_dtype=ms.float16)

    prompt = "What is your favorite condiment?"

    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
    input_ids = (
        Tensor(input_ids) if (len(input_ids.shape) == 2 and input_ids.shape[0] == 1) else Tensor(input_ids).unsqueeze(0)
    )  # (1, L)
    infer_start = time()
    # Generate
    generate_ids = model.generate(input_ids, max_length=30)
    print(f"Inference time: {time() - infer_start:.3f}s")

    print(
        tokenizer.batch_decode(
            generate_ids[:, input_ids.shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
    )


if __name__ == "__main__":
    main()
