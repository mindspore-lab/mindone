# QwQ-32B: Embracing the Power of Reinforcement Learning
[Report](https://qwenlm.github.io/blog/qwq-32b/) | [HF Model Card](https://huggingface.co/Qwen/QwQ-32B)

# Introduction
> **Abstract:** QwQ is the reasoning model of the Qwen series. Compared with conventional instruction-tuned models, QwQ, which is capable of thinking and reasoning, can achieve significantly enhanced performance in downstream tasks, especially hard problems. QwQ-32B is the medium-sized reasoning model, which is capable of achieving competitive performance against state-of-the-art reasoning models, e.g., DeepSeek-R1, o1-mini.

# Get Started

## Requirements:
|mindspore | 	ascend driver | firmware       | cann tookit/kernel|
|--- |----------------|----------------| --- |
|2.5.0 | 24.1.RC3.b080  | 7.5.T11.0.B088 | 8.0.RC3.beta1|

### Installation:
```
cd examples/transformers/qwen/qwq
pip install requirements.txt
```

Tested with:
- python==3.9.21
- mindspore==2.5.0
- transformers=4.46.3
- tokenizers==0.20.0
- mindone

Pretrained weights from huggingface hub: [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B)

## Quick Start

Here is a usage example of inference script `qwq_32B_generate.py`:

```python
import time
from functools import partial

from transformers import AutoTokenizer

import mindspore as ms
import mindspore.mint.distributed as dist
from mindspore.communication import GlobalComm

from mindone.trainers.zero import prepare_network
from mindone.transformers import Qwen2ForCausalLM

dist.init_process_group(backend="hccl")
ms.set_auto_parallel_context(parallel_mode="data_parallel")

s_time = time.time()

model_name = "Qwen/QwQ-32B"
model = Qwen2ForCausalLM.from_pretrained(model_name, mindspore_dtype=ms.bfloat16)

shard_fn = partial(prepare_network, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
model = shard_fn(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = 'How many r\'s are in the word "strawberry"'
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

input_ids = ms.Tensor(tokenizer([text], return_tensors="np").input_ids, ms.int32)
model_inputs = {}
model_inputs["input_ids"] = input_ids

generated_ids = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False, use_cache=False)

generated_ids = generated_ids.asnumpy()

outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(f"=====> input prompt: {prompt}, time cost: {time.time() - s_time:.2f}s")
print("=" * 46 + " Result " + "=" * 46)
print(outputs)
print("=" * 100)
```


To run the script, you can use the following command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
msrun --worker_num=2 --local_worker_num=2 --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 examples/qwen/qwq/qwq_32B_generate.py
```

The result is as follows, some results are omitted for brevity:
```text
Okay, so I need to figure out how many times the letter 'r' appears in the word "strawberry". Let me start by writing down the word and looking at each letter one by one.

First, I'll spell out "strawberry" to make sure I have all the letters right. S-T-R-A-W-B-E-R-R-Y. Wait, let me check that again. Sometimes I might miss a letter. Let me count the letters as I write them:

1. S
2. T
3. R
4. A
5. W
6. B
7. E
8. R
9. R
10. Y

Hmm, so that's 10 letters in total. Now, I need to count how many times 'R' shows up. Let me go through each letter again and note the positions where 'R' is.

...

Yes, that's correct. So the letters R are at position 3, 8, and 9. ...

Alternatively, maybe I can think of the pronunciation. When I say "strawberry", the first R is after the T, so "straw" has that R, and then "berrry" has two R's. So that's three.

...

THus, there are **3 r's** in "strawberry."
```
