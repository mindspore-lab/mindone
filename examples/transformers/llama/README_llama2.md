# Llama2

Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. This document will include the installation and usage of the 7B pretrained model.  Here is the [link](https://huggingface.co/meta-llama/Llama-2-7b) to its 7B huggingface weight page.

> Note:
Use of this model is governed by the Meta license. Please visit the [website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept the License before requesting access here.


The abstract of ["Llama 2: Open Foundation and Fine-Tuned Chat Models"](https://arxiv.org/abs/2307.09288) is shown below:
> In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs.


Checkout all Llama2 model checkpoints [here](https://huggingface.co/models?search=llama2).


## Get Started

### Requirements:
| mindspore   | 	ascend driver | firmware       | cann toolkit/kernel|
|-------------|----------------|----------------| --- |
| 2.6.0 | 24.1.RC3 | 7.3.0.1.231 | 8.0.RC3.beta1|

### Installation:
```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
cd examples/transformers/llama
```

## Quick Start
### Inference

```python
from transformers import AutoTokenizer
from mindone.transformers import LlamaForCausalLM
from mindspore import Tensor

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", mindspore_dtype=ms.float16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="np")

# Generate
generate_ids = model.generate(Tensor(inputs.input_ids), max_length=30, do_sample=True)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
```

The example output is show below:
```bash
Hey, are you conscious? Can you talk to me?
I am conscious. I can talk.
```

## Performance


Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.6.0.

- inference

|      model name	      | mode |  precision   | cards | flash attn | 	tokens/s	| steps|
|:---------------------:|:--:|:-----------------:|:--------------:|:---:  |:----------:|:----------:|
| Llama-2-7b-hf |  pynative| fp16 | 1 |     FALSE    |    5.18   | 16|
