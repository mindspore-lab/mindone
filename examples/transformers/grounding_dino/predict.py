import requests
from PIL import Image
from transformers import AutoProcessor

import mindspore as ms

from mindone.transformers import GroundingDinoForObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"

processor = AutoProcessor.from_pretrained(model_id)
model = GroundingDinoForObjectDetection.from_pretrained(model_id)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
# Check for cats and remote controls
text_labels = [["a cat", "a remote control"]]

inputs = processor(images=image, text=text_labels, return_tensors="np")
inputs = {k: ms.Tensor(inputs[k]) for k in inputs.keys()}
outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs, inputs["input_ids"], box_threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
)

# Retrieve the first image result
result = results[0]
for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
    box = [round(x, 2) for x in box.tolist()]
    print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
