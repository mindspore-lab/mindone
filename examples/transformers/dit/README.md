# DiT

## Introduction
[DiT](https://arxiv.org/abs/2203.02378) is an image transformer pretrained on large-scale unlabeled document images. It learns to predict the missing visual tokens from a corrupted input image. The pretrained DiT model can be used as a backbone in other models for visual document tasks like document image classification and table detection.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dit_architecture.jpg" width="800">

## Get Started

## 📦 Requirements
mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |



git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install mindone==0.4.0
```

### Checkpoints
You can find all the original DiT checkpoints under the [Microsoft](https://huggingface.co/microsoft?search_models=dit) organization. Note that you should download commit with `safetensors` variant. E.g., [dit-base-finetuned-rvlcdip safetensor commit](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip/commit/39a0713fab4fbb4a1b7785bd473c5f9708fdf8b3):
```py
commit_hash = "39a0713fab4fbb4a1b7785bd473c5f9708fdf8b3"
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/dit-base-finetuned-rvlcdip",
    revision=commit_hash
)
```

## Quick Start

The example below demonstrates how to classify an image with `pipeline` or the `AutoModel` class (omitted commit id here for simplicity).

### Pipeline
```python
import mindspore as ms
from mindone.transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="microsoft/dit-base-finetuned-rvlcdip",
    mindspore_dtype=ms.float16,
)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dit-example.jpg")
```

### AutoModel

```python
import mindspore as ms
import requests
from PIL import Image
from transformers import AutoImageProcessor
from mindone.transformers import AutoModelForImageClassification

image_processor = AutoImageProcessor.from_pretrained(
    "microsoft/dit-base-finetuned-rvlcdip",
    use_fast=True,
)
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/dit-base-finetuned-rvlcdip",
)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dit-example.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="np")
for k, v in inputs.items():
    inputs[k] = ms.tensor(v)

logits = model(**inputs).logits
predicted_class_id = logits.argmax(dim=-1).item()

class_labels = model.config.id2label
predicted_class_label = class_labels[predicted_class_id]
print(f"The predicted class label is: {predicted_class_label}")
# The predicted class label is: advertisement
```
### Notes
The pretrained DiT weights can be loaded in a [BEiT](../../../mindone/transformers/models/beit/modeling_beit.py) model with a modeling head to predict visual tokens.
```py
from mindone.transformers import BeitForMaskedImageModeling
model = BeitForMaskedImageModeling.from_pretrained("microsoft/dit-base")
```

## Inference Speed

Experiments are tested on ascend 910* with mindspore 2.7.0 pynative mode.
|      model name	      | use pipeline |   precision   | cards | flash attn | 	s/step	 |
|:---------------------:|:-----------------:|:--------------:|:---:  |:----------:|:----------:|
| microsoft/dit-base-finetuned-rvlcdip |  TRUE    |  fp32 | 1 |     OFF      |    0.306    |
| microsoft/dit-base-finetuned-rvlcdip |  FALSE   |  fp32 | 1 |     OFF      |    0.025    |
