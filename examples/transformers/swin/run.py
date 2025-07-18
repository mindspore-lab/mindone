# Adapted from https://huggingface.co/docs/transformers/model_doc/swin#transformers.TFSwinModel.call.example
import requests
from PIL import Image

import mindspore as ms

from mindone.transformers import AutoBackbone, AutoImageProcessor

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
model = AutoBackbone.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224", out_features=["stage1", "stage2", "stage3", "stage4"]
)

inputs = ms.Tensor(processor(image, return_tensors="np")["pixel_values"])
outputs = model(pixel_values=inputs)
feature_maps = outputs.feature_maps
