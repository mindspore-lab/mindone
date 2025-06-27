# Idefics3

# Introduction
The Idefics3 model was proposed in Building and better understanding vision-language models: insights and future directions by Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon.

Idefics3 is an adaptation of the Idefics2 model with three main differences:

- It uses Llama3 for the text model.
- It uses an updated processing logic for the images.
- It removes the perceiver.

The abstract from the paper is the following:

> The field of vision-language models (VLMs), which take images and texts as inputs and output texts, is rapidly evolving and has yet to reach consensus on several key aspects of the development pipeline, including data, architecture, and training methods. This paper can be seen as a tutorial for building a VLM. We begin by providing a comprehensive overview of the current state-of-the-art approaches, highlighting the strengths and weaknesses of each, addressing the major challenges in the field, and suggesting promising research directions for underexplored areas. We then walk through the practical steps to build Idefics3-8B, a powerful VLM that significantly outperforms its predecessor Idefics2-8B, while being trained efficiently, exclusively on open datasets, and using a straightforward pipeline. These steps include the creation of Docmatix, a dataset for improving document understanding capabilities, which is 240 times larger than previously available datasets. We release the model along with the datasets created for its training.

# Get Started

## Requirements:
|mindspore | ascend driver | firmware | cann tookit/kernel|
|--- |--- | ---| --- |
|2.5.0 | 24.1.RC3 | 7.5.T11.0 | 8.0.0.beta1|
|2.6.0 | 24.1.RC3 | 7.5.T11.0 | 8.0.0.beta1|

## Quick Start

### Installation:
```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
cd examples/transformers/idefics3
```

### Usage example
You can use the following script to run multi-image VQA:

```bash
python generate.py
```

## Inference Speed

Experiments are tested on ascend 910* with pynative mode.

- mindspore 2.5.0

|model name	| precision | cards | fa  |	tokens/s	| weight |
| :---: | :---:  |:---:  | :---:  |:---:  | ---|
| Idefics3-8B-Llama3 |  fp16 | 1 | ON  | 3.86 | [weight](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)
| Idefics3-8B-Llama3 |  bf16 | 1 | ON  | 4.22 | [weight](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)

- mindspore 2.6.0

|model name	| precision | cards | fa  |	tokens/s	| weight |
| :---: | :---:  |:---:  | :---:  |:---:  | ---|
| Idefics3-8B-Llama3 |  fp16 | 1 | ON  | 4.16 | [weight](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)
| Idefics3-8B-Llama3 |  bf16 | 1 | ON  | 4.53  | [weight](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)
