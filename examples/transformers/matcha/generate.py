import requests
from PIL import Image
from transformers import AutoProcessor

import mindspore as ms

from mindone.transformers import Pix2StructForConditionalGeneration

model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-chartqa")
processor = AutoProcessor.from_pretrained("google/matcha-chartqa")
url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/20294671002019.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="Is the sum of all 4 places greater than Laos?", return_tensors="np")
inputs = {k: ms.Tensor(v) for k, v in inputs.items()}
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))  # No
