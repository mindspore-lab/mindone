import time
from functools import partial

from transformers import AutoTokenizer

from mindone.trainers.zero import prepare_network
from mindone.transformers import Qwen2ForCausalLM
import mindspore as ms
import mindspore.mint.distributed as dist

from mindspore.communication import GlobalComm

dist.init_process_group(backend="hccl")
ms.set_auto_parallel_context(parallel_mode="data_parallel")

s_time = time.time()

model_name = "Qwen/QwQ-32B"
model = Qwen2ForCausalLM.from_pretrained(model_name, mindspore_dtype=ms.bfloat16)

shard_fn = partial(prepare_network, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
model = shard_fn(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "How many r's are in the word \"strawberry\""
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

input_ids = ms.Tensor(tokenizer([text], return_tensors="np").input_ids, ms.int32)
model_inputs = {}
model_inputs["input_ids"] = input_ids

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
    do_sample=False,
    use_cache=False
)

generated_ids = generated_ids.asnumpy()

outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(f"=====> input prompt: {prompt}, time cost: {time.time() - s_time:.2f}s")
print("=" * 46 + " Result " + "=" * 46)
print(outputs)
print("=" * 100)