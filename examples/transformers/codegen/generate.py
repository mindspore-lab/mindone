from transformers import AutoTokenizer

import mindspore as ms

from mindway.transformers.models.codegen import CodeGenForCausalLM

checkpoint = "Salesforce/codegen-350M-mono"
model = CodeGenForCausalLM.from_pretrained(checkpoint, use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = "def hello_world():"
completion = model.generate(input_ids=ms.tensor(tokenizer(text, return_tensors="np").input_ids, dtype=ms.int32))
print(f"result: {tokenizer.decode(completion[0])}")
