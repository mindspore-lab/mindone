from moviegen.models import llama3_1B, llama3_5B, llama3_30B

import mindspore as ms

__all__ = ["MODEL_SPEC", "MODEL_DTYPE"]

MODEL_SPEC = {"llama-1B": llama3_1B, "llama-5B": llama3_5B, "llama-30B": llama3_30B}

MODEL_DTYPE = {
    "fp32": ms.float32,
    "fp16": ms.float16,
    "bf16": ms.bfloat16,
}
