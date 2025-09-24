# Qwen3-VL series

## Introduction
[Qwen3-VL](https://huggingface.co/papers/2502.13923) is a multimodal vision-language model series, encompassing both dense and MoE variants, as well as Instruct and Thinking versions. Building upon its predecessors, Qwen3-VL delivers significant improvements in visual understanding while maintaining strong pure text capabilities. Key architectural advancements include: enhanced MRope with interleaved layout for better spatial-temporal modeling, DeepStack integration to effectively leverage multi-level features from the Vision Transformer (ViT), and improved video understanding through text-based time alignment—evolving from T-RoPE to text timestamp alignment for more precise temporal grounding. These innovations collectively enable Qwen3-VL to achieve superior performance in complex multimodal tasks.

# Get Started

## Requirements:
| mindspore | 	ascend driver | firmware       | cann tookit/kernel |
|-----------|----------------|----------------|--------------------|
| 2.6.0     | 24.1.RC3.b080  | 7.5.T11.0.B088 | 8.1.RC1            |

### Installation:
```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
cd examples/transformers/glm4v

# compile newest transformers whl because qwen3-vl(transformers v4.57.dev.0) haven't released
git clone https://github.com/huggingface/transformers.git
pip install -e .
```

## Quick Start

Here is a usage example of Qwen3-VL-4B-Instruct. you can use the following command:

```bash
# for Qwen3-VL-4B-Instruct inference
python generate.py
    --model_name "Qwen/Qwen3-VL-4B-Instruct"
    --image "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    --prompt "Describe this image."
```

```bash
# for Qwen3-VL-30B-Instruct inference
msrun --worker_num=2 --local_worker_num=2 master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 generate.py
    --model_name "Qwen/Qwen3-VL-30B-Instruct"
    --image "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    --prompt "Describe this image."
```

Image:
![sample image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg)

Prompt: Describe this image.

Qwen3-VL-4B Outputs:
```
['Of course, here is detailed description of the image provided.\n\n
 This is a close-up photograph of a Pallas\'s cat ($Felis$, $manul$),
an endangered wild feline species native to Central Aisa.
...
**Appearance:** It has a stocky and robust build with short legs
and a large head relative to its body size. Its fur is thick and dense,
appearing somewhat fluffy or "matted,", which is characteristic']
```

Qwen3-VL-30B Outputs:
```
['Of course, here is detailed description of the image provided.\n\n
This is a dynamic and charming photograph of a Palla's cat (also known as a manul) in a snowy enviroment.
...
"Appearance:" The cat has a very distinctive apperance, characterized by its stocky, low-slung body and exceptionally
thick, dense fur. This coat is a mix of brownish"]
```

`model_name` and `image` could be replaced with your local path. Give it a try with various images and prompts🤗🤗.

## Inference Speed
|        model name	         | mindspore version | precision* | cards | attention type | 	tokens/s	 |
|:--------------------------:|:-----------------:|:----------:|:---:  |:--------------:|:----------:|
| Qwen/Qwen3-VL-4B-Instruct  |       2.6.0       |    bf16     | 1 |   flash_attn   |    1.35    |
| Qwen/Qwen3-VL-30B-Instruct |       2.6.0       |    bf16    | 1 |   flash_attn   |    0.5     |

