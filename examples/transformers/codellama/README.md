
# CodeLlama

## Overview

The Code Llama models were proposed in [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/).

All Code Llama model checkpoints can be found [here](https://huggingface.co/models?search=code_llama), and the officially released checkpoints at [meta llama org](https://huggingface.co/meta-llama).

This model was contributed by [ArthurZucker](https://huggingface.co/ArthurZ). The original code can be found [here](https://github.com/facebookresearch/llama).

## Checkpoints

CodeLlama checkpoints are restricted. One need to authenticate with Hugging Face to download the checkpoints. We recommend using the following command to authenticate:

```bash
huggingface-cli login
```
Login with your HuggingFace access token with the correct permissions.

Afterwards, you can download the checkpoints using the following command:
```bash
huggingface-cli download --resume-download meta-llama/CodeLlama-7b-hf
```

## Examples

Here's an example usage:

```bash
>>> from transformers import CodeLlamaTokenizer
>>> from mindone.transformers.models.llama import LlamaForCausalLM
>>> import mindspore as ms

>>> tokenizer = CodeLlamaTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
>>> model = LlamaForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf", use_flash_attention_2=True, mindspore_dtype=ms.float16) # model weight will be automatically downloaded from huggingface
>>> PROMPT = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''
>>> input_ids = ms.Tensor(tokenizer(prompt, return_tensors="np").input_ids, ms.int32)
>>> generated_ids = model.generate(input_ids, max_new_tokens=128,  do_sample=False).asnumpy()

>>> filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
>>> print(PROMPT.replace("<FILL_ME>", filling))
def remove_non_ascii(s: str) -> str:
    """ Remove non-ASCII characters from a string.

    Args:
        s: The string to remove non-ASCII characters from.

    Returns:
        The string with non-ASCII characters removed.
    """
    result = ""
    for c in s:
        if ord(c) < 128:
            result += c
    return result
```
Internally, the tokenizer automatically splits by <FILL_ME> to create a formatted input string following the original training pattern. This is more robust than preparing the pattern yourself as it avoids very hard-to-debug pitfalls like token glueing.

The LLaMA tokenizer is a BPE model based on sentencepiece. One quirk of sentencepiece is that when decoding a sequence, if the first token is the start of a word (e.g., "Banana"), the tokenizer does not prepend the prefix space to the string.

Code Llama has the same architecture as the Llama2 models. For API reference, see the Llama2 documentation page.
