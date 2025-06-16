import requests
from PIL import Image

import mindspore as ms
from mindspore import mint

from mindone.transformers import AutoImageProcessor, SwinForMaskedImageModeling

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-simmim-window6-192")
model = SwinForMaskedImageModeling.from_pretrained("microsoft/swin-base-simmim-window6-192")

num_patches = (model.config.image_size // model.config.patch_size) ** 2
pixel_values = ms.Tensor(image_processor(images=image, return_tensors="np").pixel_values)
# create random boolean mask of shape (batch_size, num_patches)
bool_masked_pos = mint.randint(low=0, high=2, size=(1, num_patches)).bool()

outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
loss, reconstructed_pixel_values = outputs.loss, outputs.logits
print(list(reconstructed_pixel_values.shape))
