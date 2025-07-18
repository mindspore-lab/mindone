# Adapted from https://huggingface.co/docs/transformers/model_doc/swin#transformers.TFSwinModel.call.example
import requests
from PIL import Image

import mindspore as ms

from mindone.transformers import AutoImageProcessor, SwinModel

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

inputs = ms.Tensor(processor(image, return_tensors="np")["pixel_values"])
outputs = model(pixel_values=inputs)
last_hidden_states = outputs.last_hidden_state
