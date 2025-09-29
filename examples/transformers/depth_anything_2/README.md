# Depth Anything V2

## Introduction
Depth Anything V2 was introduced in the [paper of the same name](https://arxiv.org/abs/2406.09414) by Lihe Yang et al. It uses the **same architecture as the original [Depth Anything](https://arxiv.org/abs/2401.10891) model**, but uses synthetic data and a larger capacity teacher model to achieve much finer and robust depth predictions.

The abstract from the paper is the following:

> This work presents Depth Anything V2. Without pursuing fancy techniques, we aim to reveal crucial findings to pave the way towards building a powerful monocular depth estimation model. Notably, compared with V1, this version produces much finer and more robust depth predictions through three key practices: 1) replacing all labeled real images with synthetic images, 2) scaling up the capacity of our teacher model, and 3) teaching student models via the bridge of large-scale pseudo-labeled real images. Compared with the latest models built on Stable Diffusion, our models are significantly more efficient (more than 10x faster) and more accurate. We offer models of different scales (ranging from 25M to 1.3B params) to support extensive scenarios. Benefiting from their strong generalization capability, we fine-tune them with metric depth labels to obtain our metric depth models. In addition to our models, considering the limited diversity and frequent noise in current test sets, we construct a versatile evaluation benchmark with precise annotations and diverse scenes to facilitate future research.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/depth_anything_overview.jpg" width="800">

*Depth Anything overview. Taken from the [original paper](https://arxiv.org/abs/2406.09414).*

## Get Started

### Requirements:
| mindspore   | 	ascend driver | firmware       | cann toolkit/kernel|
|-------------|----------------|----------------| --- |
| 2.6.0|  24.1.RC3 | 7.5.T11.0 | 8.1.RC1|

### Installation:
```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
```

## Quick Start

There are 2 main ways to use Depth Anything V2: either using the pipeline API, which abstracts away all the complexity for you, or by using the `DepthAnythingForDepthEstimation` class yourself.

### Pipeline API

The pipeline allows to use the model in a few lines of code:
```python
from mindone.transformers import pipeline
from PIL import Image
import requests

# load pipe
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# inference
depth = pipe(image)["depth"]
depth.save("depth.jpg")
```

### Using the model yourself
If you want to do the pre- and post-processing yourself, here’s how to do that:
```python
from mindone.transformers import AutoImageProcessor, AutoModelForDepthEstimation
import mindspore as ms
import numpy as np
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="ms")

outputs = model(**inputs)

# interpolate to original size and visualize the prediction
post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    target_sizes=[(image.height, image.width)],
)

predicted_depth = post_processed_output[0]["predicted_depth"]
depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
depth = depth.asnumpy() * 255
depth = Image.fromarray(depth.astype("uint8"))
depth.save("depth.jpg")
```

## Inference Speed

Experiments are tested on ascend 910* with mindspore 2.6.0 pynative mode.
|      model name	      | use pipeline |   precision   | cards | flash attn | 	s/step	 |
|:---------------------:|:-----------------:|:--------------:|:---:  |:----------:|:----------:|
| depth-anything/Depth-Anything-V2-Small-hf |  TRUE    |  fp32 | 1 |     OFF      |    0.062    |
| depth-anything/Depth-Anything-V2-Small-hf |  FALSE   |  fp32 | 1 |     OFF      |    0.028    |
