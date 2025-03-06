# CogView4

## News

- 🔥🔥 ```2025/03/05```: We have reproduced the inference of the excellent work CogView4, which was open-sourced by THUDM, on MindSpore.


## Introduction

### Model Download

| Models   | 🤗Huggingface    |
|:-------:|:-------:|
| CogView4 | [download](https://huggingface.co/THUDM/CogView4-6B)|

### Dependencies and Installation

| mindspore  | ascend driver  |  firmware   | cann toolkit/kernel |
|:----------:|:--------------:|:-----------:|:------------------:|
|   2.5.0    |     24.1RC2    | 7.3.0.1.231 |   8.0.0.beta1    |

To install other dependent packages:

```
git clone https://github.com/mindspore-lab/mindone

# install mindone
cd mindone
pip install -e .
# NOTE: transformers requires >=4.46.0

cd examples/cogview
```


## Quick Start

### Using Command Line

We provide the command to quick start:

```shell
cd inference
python cli_demo_cogview4.py --prompt {your prompt}
```

### Using Diffusers

Run the model with `BF16` precision:

```python
from mindone.diffusers import CogView4Pipeline
import mindspore as ms

pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", mindspore_dtype=ms.bfloat16)

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."
image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    num_images_per_prompt=1,
    num_inference_steps=50,
    width=1024,
    height=1024,
)[0][0]

image.save("cogview4.png")
```

### Performance

#### Inference

|model|resolution|50 steps|
|:----------:|:--------------:|:-----------:|
|THUDM/CogView4-6B|720*1280|40s|
|THUDM/CogView4-6B|512*512|30s|
|THUDM/CogView4-6B|1024*1024|45s|

prompt: 一面涂鸦墙，墙上写着涂鸦字体的英文“MindOne”
<p align="left"><img width="512" src="https://github.com/user-attachments/assets/ecedad99-a0d7-4428-80aa-db43175030ec"/></p>
