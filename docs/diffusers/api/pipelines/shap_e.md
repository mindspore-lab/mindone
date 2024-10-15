<!--Copyright 2024 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Shap-E

The Shap-E model was proposed in [Shap-E: Generating Conditional 3D Implicit Functions](https://huggingface.co/papers/2305.02463) by Alex Nichol and Heewoo Jun from [OpenAI](https://github.com/openai).

The abstract from the paper is:

*We present Shap-E, a conditional generative model for 3D assets. Unlike recent work on 3D generative models which produce a single output representation, Shap-E directly generates the parameters of implicit functions that can be rendered as both textured meshes and neural radiance fields. We train Shap-E in two stages: first, we train an encoder that deterministically maps 3D assets into the parameters of an implicit function; second, we train a conditional diffusion model on outputs of the encoder. When trained on a large dataset of paired 3D and text data, our resulting models are capable of generating complex and diverse 3D assets in a matter of seconds. When compared to Point-E, an explicit generative model over point clouds, Shap-E converges faster and reaches comparable or better sample quality despite modeling a higher-dimensional, multi-representation output space.*

The original codebase can be found at [openai/shap-e](https://github.com/openai/shap-e).

!!! tip

	See the [reuse components across pipelines](../../using-diffusers/loading.md) section to learn how to efficiently load the same components into multiple pipelines.

::: mindone.diffusers.ShapEPipeline

::: mindone.diffusers.ShapEImg2ImgPipeline

::: mindone.diffusers.pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput

## Usage

````
# ShapEPipeline
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import export_to_gif

repo = "openai/shap-e"
pipe = DiffusionPipeline.from_pretrained(repo, mindspore_dtype=ms.float16)

guidance_scale = 15.0
prompt = "a shark"

images = pipe(
     prompt,
     guidance_scale=guidance_scale,
     num_inference_steps=64,
     frame_size=256,
 )[0]

export_to_gif(images[0], "shark_3d.gif")
````

````
# ShapEImg2ImgPipeline
import mindspore as ms
from mindone.diffusers.pipelines import ShapEImg2ImgPipeline
from mindone.diffusers.utils import export_to_gif, load_image

repo = "openai/shap-e-img2img"
pipe = ShapEImg2ImgPipeline.from_pretrained(repo, mindspore_dtype=ms.float16)

guidance_scale = 3.0
image_url = "corgi.png"
image = load_image(image_url).convert("RGB")

images = pipe(
     image,
     guidance_scale=guidance_scale,
     num_inference_steps=64,
     frame_size=256,
     )[0]

gif_path = export_to_gif(images[0], "corgi_3d.gif")
````
