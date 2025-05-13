from transformers import AutoTokenizer

import mindspore as ms

from mindone.transformers import Qwen2ForCausalLM

ms.set_context(mode=0)

model_name = "/mnt/disk2/wcr/Qwen2.5-14B-Instruct"
model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    mindspore_dtype=ms.bfloat16,
    attn_implementation="paged_attention",
)

# infer boost
from mindspore import JitConfig

jitconfig = JitConfig(jit_level="O0", infer_boost="on")
model.set_jit_config(jitconfig)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

input_ids = ms.Tensor(tokenizer([text], return_tensors="np").input_ids, ms.int32)
model_inputs = {}
model_inputs["input_ids"] = input_ids

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=False,
    use_cache=False,
)

generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(input_ids, generated_ids)]

outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(outputs)
