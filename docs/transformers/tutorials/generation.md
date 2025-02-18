<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Generation with LLMs

LLMs, or Large Language Models, are the key component behind text generation. In a nutshell, they consist of large pretrained transformer models trained to predict the next word (or, more precisely, token) given some input text. Since they predict one token at a time, you need to do something more elaborate to generate new sentences other than just calling the model — you need to do autoregressive generation.

Autoregressive generation is the inference-time procedure of iteratively calling a model with its own generated outputs, given a few initial inputs. In 🤗 Transformers, this is handled by the generate() method, which is available to all models with generative capabilities.

This tutorial will show you how to:

- Generate text with an LLM

Before you begin, make sure you have all the necessary libraries installed:

```shell
pip install transformers==4.42.4
```

## Generate text

!!! Note

    Taking llama as an example, you can find the complete code in `examples/transformers/llama/generate.py`
    And you can compare the results of script `examples/transformers/llama/generate_pt.py` with PyTorch.

A language model trained for causal language modeling takes a sequence of text tokens as input and returns the probability distribution for the next token.

A critical aspect of autoregressive generation with LLMs is how to select the next token from this probability distribution. Anything goes in this step as long as you end up with a token for the next iteration. This means it can be as simple as selecting the most likely token from the probability distribution or as complex as applying a dozen transformations before sampling from the resulting distribution.

The process depicted above is repeated iteratively until some stopping condition is reached. Ideally, the stopping condition is dictated by the model, which should learn when to output an end-of-sequence (EOS) token. If this is not the case, generation stops when some predefined maximum length is reached.

Properly setting up the token selection step and the stopping condition is essential to make your model behave as you’d expect on your task. That is why we have a GenerationConfig file associated with each model, which contains a good default generative parameterization and is loaded alongside your model.

Let’s talk code!

!!! Note

    If you’re interested in basic LLM usage, our high-level Pipeline interface is a great starting point. However, LLMs often require advanced features like quantization and fine control of the token selection step, which is best done through generate(). Autoregressive generation with LLMs is also resource-intensive and should be executed on a Ascend NPU for adequate throughput.

First, you need to load the model.

```pycon
>>> from mindone.transformers.models.llama import LlamaForCausalLM

>>> model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
```

There are other ways to initialize a model, but this is a good baseline to begin with an LLM.

Next, you need to preprocess your text input with a tokenizer.

```pycon
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
>>> input_ids = ms.Tensor(tokenizer(["A list of colors: red, blue"]).input_ids, ms.int32)
```

The model_inputs variable holds the tokenized text input, as well as the attention mask. While generate() does its best effort to infer the attention mask when it is not passed, we recommend passing it whenever possible for optimal results.

After tokenizing the inputs, you can call the generate() method to returns the generated tokens. The generated tokens then should be converted to text before printing.

```pycon
>>> generated_ids = model.generate(
...     input_ids=input_ids,
...     max_new_tokens=30,
...     use_cache=True,
...     do_sample=False,
... )

>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
```

Finally, you don’t need to do it one sequence at a time! You can batch your inputs, which will greatly improve the throughput at a small latency and memory cost. All you need to do is to make sure you pad your inputs properly (more on that below).

```pycon
>>> tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
>>> input_ids = ms.Tensor(tokenizer(
...     ["A list of colors: red, blue", "Portugal is"], padding=True
... ).input_ids, ms.int32)

>>> generated_ids = model.generate(
...     input_ids=input_ids,
...     max_new_tokens=30,
...     use_cache=True,
...     do_sample=False,
... )

>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

And that’s it! In a few lines of code, you can harness the power of an LLM.
