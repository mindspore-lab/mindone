import requests
from PIL import Image
from transformers import AutoProcessor

import mindspore as ms

from mindone.transformers import Pix2StructForConditionalGeneration

model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
processor = AutoProcessor.from_pretrained("google/deplot")
url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="np")
inputs = {k: ms.Tensor(v) for k, v in inputs.items()}
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
