# GLM-4.1V

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
cd examples/transformers/glm4v
```

## Quick Start

Here is a usage example of GLM-4.1V-Thinking. you can use the following command:

```bash
# for GLM-4.1V-Thinking inference
python generate.py
    --model_name "THUDM/GLM-4.1V-9B-Thinking"
    --image "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
```

## Inference Speed
|      model name	      | precision* | cards | flash attn | 	tokens/s	 |
|:---------------------:| :---:  |:---:  |:----------:|:----------:|
| THUDM/GLM-4.1V-9B-Thinking |  bf16 | 1 |     ✅      |    1.38    |
