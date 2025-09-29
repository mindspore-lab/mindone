# LLama3


## Introduction

The Llama3 model was proposed in Introducing [Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/) by the meta AI team.

The abstract from the blogpost is the following:

> Today, we’re excited to share the first two models of the next generation of Llama, Meta Llama 3, available for broad use. This release features pretrained and instruction-fine-tuned language models with 8B and 70B parameters that can support a broad range of use cases. This next generation of Llama demonstrates state-of-the-art performance on a wide range of industry benchmarks and offers new capabilities, including improved reasoning. We believe these are the best open source models of their class, period. In support of our longstanding open approach, we’re putting Llama 3 in the hands of the community. We want to kickstart the next wave of innovation in AI across the stack—from applications to developer tools to evals to inference optimizations and more. We can’t wait to see what you build and look forward to your feedback.

Checkout all Llama3 model checkpoints [here](https://huggingface.co/models?search=llama3).


## Get Started

### Requirements:
| mindspore   | 	ascend driver | firmware       | cann toolkit/kernel|
|-------------|----------------|----------------| --- |
|2.5.0 | 24.1.RC3 | 7.3.0.1.231 | 8.0.RC3.beta1|

### Installation:
```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
cd examples/transformers/llama
```

## Quick Start
### Inference
You can run the script `generate.py` for inference, it supports pynative or graph mode.
```bash
DEVICE_ID=0 python generate.py \
--ms_mode 1 \
--model_path meta-llama/Llama-3-8B
```
Then you can get the hints to enter your prompt: "Enter your prompt [e.g. \`What's your name?\`] or enter [\`q\`] to exit: "

Here is an example of input and response:
```
Enter your prompt [e.g. `What's your name?`] or enter [`q`] to exit: Where is China?
China is located in eastern part of Asia. It is the world's most populous country. China is the world's second
```

# Training
You can run the training script `finetune_in_native_mindspore.py` or `finetune_with_mindspore_trainer.py` for finetuning in graph mode, here is an example:
```bash
DEVICE_ID=0 python finetune_in_native_mindspore.py \
--model_path meta-llama/Llama-3-8B
```

## Performance


Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.5.0.

- inference

|      model name	      | mode |  precision   | cards | flash attn | 	tokens/s	| steps|
|:---------------------:|:--:|:-----------------:|:--------------:|:---:  |:----------:|:----------:|
| Llama3-8B |  pynative| fp16 | 1 |     ON    |    1.59    |23|
| Llama3-8B |  graph| fp16 | 1 |     ON    |    8.51    |23|
| Llama3.2-1B | pynative| fp16 | 1 |   ON     |    3.02    |24|
| Llama3.2-1B | graph| fp16 | 1 |   ON     |    16.62   |24|

- finetune

|      model name	      | mode |  graph compile| precision  | recompute | zero |jit level | cards | batch | flash attn | 	it/s	|
|:---------------------:|:--:|:--:|:--:|:-----------------:|:--------------:|:---: |:---:|:---:  |:----------:|:----------:|
| Llama3.2-1B | graph(trainer)| 3 mins| fp16 | ON |0| O0| 1 | 8 |   ON     |   1.77|
| Llama3.2-1B | graph(native ms)| 3 mins |fp16 |ON |0| O0|  1 | 8|  ON     |   |
