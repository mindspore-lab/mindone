
# Cohere2

## Overview

[C4AI Command R7B](https://cohere.com/blog/command-r7b) is an open weights research release of a 7B billion parameter model developed by Cohere and Cohere For AI. It has advanced capabilities optimized for various use cases, including reasoning, summarization, question answering, and code. The model is trained to perform sophisticated tasks including Retrieval Augmented Generation (RAG) and tool use. The model also has powerful agentic capabilities that can use and combine multiple tools over multiple steps to accomplish more difficult tasks. It obtains top performance on enterprise-relevant code use cases. C4AI Command R7B is a multilingual model trained on 23 languages.

The model features three layers with sliding window attention (window size 4096) and ROPE for efficient local context modeling and relative positional encoding. A fourth layer uses global attention without positional embeddings, enabling unrestricted token interactions across the entire sequence.

The model has been trained on 23 languages: English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Arabic, Chinese, Russian, Polish, Turkish, Vietnamese, Dutch, Czech, Indonesian, Ukrainian, Romanian, Greek, Hindi, Hebrew, and Persian.



## Checkpoints

You can download the checkpoints using the following command:
```bash
huggingface-cli download --resume-download CohereForAI/c4ai-command-r7b-12-2024
```

## Examples

Here's an example usage:

```python
from time import time

from transformers import AutoTokenizer

import mindspore as ms
from mindspore import Tensor

from mindone.transformers import Cohere2ForCausalLM

model_id = "CohereForAI/c4ai-command-r7b-12-2024"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Cohere2ForCausalLM.from_pretrained(model_id, mindspore_dtype=ms.float16)

messages = [{"role": "user", "content": "How do plants make energy?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="np")

input_ids = (
    Tensor(input_ids) if (len(input_ids.shape) == 2 and input_ids.shape[0] == 1) else Tensor(input_ids).unsqueeze(0)
)  # (1, L)
infer_start = time()
output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    cache_implementation="static",
)
print(f"Inference time: {time() - infer_start:.3f}s")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

See `./generate.py` for detailed usage.
