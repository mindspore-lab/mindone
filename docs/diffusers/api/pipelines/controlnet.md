<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNet

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

With a ControlNet model, you can provide an additional control image to condition and control Stable Diffusion generation. For example, if you provide a depth map, the ControlNet model generates an image that'll preserve the spatial information from the depth map. It is a more flexible and accurate way to control the image generation process.

The abstract from the paper is:

*We present ControlNet, a neural network architecture to add spatial conditioning controls to large, pretrained text-to-image diffusion models. ControlNet locks the production-ready large diffusion models, and reuses their deep and robust encoding layers pretrained with billions of images as a strong backbone to learn a diverse set of conditional controls. The neural architecture is connected with "zero convolutions" (zero-initialized convolution layers) that progressively grow the parameters from zero and ensure that no harmful noise could affect the finetuning. We test various conditioning controls, eg, edges, depth, segmentation, human pose, etc, with Stable Diffusion, using single or multiple conditions, with or without prompts. We show that the training of ControlNets is robust with small (<50k) and large (>1m) datasets. Extensive results show that ControlNet may facilitate wider applications to control image diffusion models.*

This model was contributed by [takuma104](https://huggingface.co/takuma104). ❤️

The original codebase can be found at [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet), and you can find official ControlNet checkpoints on [lllyasviel's](https://huggingface.co/lllyasviel) Hub profile.

!!! tip

	Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md) section to learn how to efficiently load the same components into multiple pipelines.


## Usage Example

```python
import cv2
import numpy as np
import mindspore as ms
from PIL import Image
from mindone.diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from mindone.diffusers.utils import load_image


image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
np_image = np.array(image)

np_image = cv2.canny(np_image, 100, 200)
np_image = np_image[:,:,None]
np_image = np.concatenate([np_image, np_image, np_image], axis=2)
canny_image = Image.fromarray(np_image)

controlnet = ControlNetModel.from_pretrained(
	"lllyasviel/sd-controlnet-canny",
	mindspore_dtype=ms.float16,
	use_safetensors=True
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
	"runwayml/stable-diffusion-v1-5",
	controlnet = controlnet,
	mindspore_dtype = ms.float16,
	use_safetensors = True,
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

image = pipe(
	"futuristic-looking woman",
	num_inference_steps=20,
	image=image,
	control_image=canny_image,
)[0][0]
image.save("image.png")
```

::: mindone.diffusers.StableDiffusionControlNetPipeline

::: mindone.diffusers.StableDiffusionControlNetImg2ImgPipeline

::: mindone.diffusers.StableDiffusionControlNetInpaintPipeline

::: mindone.diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput
