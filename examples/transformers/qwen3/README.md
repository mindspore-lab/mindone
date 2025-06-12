# Qwen3

# Introduction
Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and MoE models.  
The advancements of Qwen3 series are as follows:  
1. support seamless switching between thinking mode and non-thinking mode
2. significantly enhancement in its reasoning ability compared with previous QwQ and Qwen 2.5 Model
3. superior human prefrence alignment, excelling in creative-writing, role-playing, multi-turn dialogues and etc
3. expertise in agent capabilities
4. Support multiple languages

Model structure is also evolving. RMSNorm for Q and K have been added in attention layer to reduce variance.
Besides that, Qwen3 apply normalization in each head. Finally, shared postion embeddings have been applied.

# Get Started

## Requirements:
|mindspore | 	ascend driver | firmware       | cann tookit/kernel|
|--- |----------------|----------------| --- |
|2.5.0 | 24.1.RC3.b080  | 7.5.T11.0.B088 | 8.0.RC3.beta1|

### Installation:
```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
cd examples/qwen3
```

## Quick Start

Here is a usage example of inference script:

```python
import mindspore
from mindspore import JitConfig
from transformers import AutoTokenizer
from mindone.transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_syntax_level=mindspore.STRICT)

# load model
model_name = "Qwen/Qwen3-0.6B"  # or replace the local path here
model = Qwen3ForCausalLM.from_pretrained(
    model_name,
    mindspore_dtype=mindspore.bfloat16,
    attn_implementation="paged_attention",
)

jitconfig = JitConfig(jit_level="O0", infer_boost="on")
model.set_jit_config(jitconfig)
config = model.config
tokenizer = AutoTokenizer.from_pretrained(model_name)

# info
print("*" * 100)
print(f"Using {config._attn_implementation}, use_cache {config.use_cache},"
      f"dtype {config.mindspore_dtype}, layer {config.num_hidden_layers}")
print("Successfully loaded Qwen3ForCausalLM")

# prepare inputs
prompt = "the secret to baking a really good cake is"
input_ids = mindspore.tensor(tokenizer([prompt], return_tensors="np").input_ids, mindspore.int32)
model_inputs = {}
model_inputs["input_ids"] = input_ids

# generate
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=50,
    do_sample=False,
    use_cache=False,
)

generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)]
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(outputs)
```

For convienience, you can use the following command:

```bash
python generate.py \
    --model_name "Qwen/Qwen3-0.6B-Base" \
    --prompt "the secret to baking a really good cake is"
```

## Inference Speed
|model name	| precision* | cards | page attn |	tokens/s	|
| :---: | :---:  |:---:  | :---:  |:---:  |
| qwen3-0.6B-base |  bf16 | 1 | ✅  | 20.33 |
| qwen3-0.6B-0424 |  bf16 | 1 | ✅  | 21.13 |
| qwen3-1.7B-base |  bf16 | 1 | ✅  | 21.23 |
| qwen3-0.6B-0424 |  bf16 | 1 | ✅  | 22.85 |
| qwen3-4B-base |  bf16 | 1 | ✅  | 22.24 |
| qwen3-4B-0426 |  bf16 | 1 | ✅  | 19.92 |
| qwen3-8B-base |  bf16 | 1 | ✅  | 19.48 |
| qwen3-8B-0424 |  bf16 | 1 | ✅  | 19.12 |
