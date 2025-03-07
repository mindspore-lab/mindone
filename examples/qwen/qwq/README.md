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

Here is a usage example of inference:

```python
TODO
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

