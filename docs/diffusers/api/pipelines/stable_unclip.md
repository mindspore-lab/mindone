<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable unCLIP

Stable unCLIP checkpoints are finetuned from [Stable Diffusion 2.1](./stable_diffusion/stable_diffusion_2.md) checkpoints to condition on CLIP image embeddings.
Stable unCLIP still conditions on text embeddings. Given the two separate conditionings, stable unCLIP can be used
for text guided image variation. When combined with an unCLIP prior, it can also be used for full text to image generation.

The abstract from the paper is:

*Contrastive models like CLIP have been shown to learn robust representations of images that capture both semantics and style. To leverage these representations for image generation, we propose a two-stage model: a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding. We show that explicitly generating image representations improves image diversity with minimal loss in photorealism and caption similarity. Our decoders conditioned on image representations can also produce variations of an image that preserve both its semantics and style, while varying the non-essential details absent from the image representation. Moreover, the joint embedding space of CLIP enables language-guided image manipulations in a zero-shot fashion. We use diffusion models for the decoder and experiment with both autoregressive and diffusion models for the prior, finding that the latter are computationally more efficient and produce higher-quality samples.*

## Tips

Stable unCLIP takes  `noise_level` as input during inference which determines how much noise is added to the image embeddings. A higher `noise_level` increases variation in the final un-noised images. By default, we do not add any additional noise to the image embeddings (`noise_level = 0`).

### Text-to-Image Generation
Stable unCLIP can be leveraged for text-to-image generation by pipelining it with the prior model of KakaoBrain's open source DALL-E 2 replication [Karlo](https://huggingface.co/kakaobrain/karlo-v1-alpha):

```python
import mindspore as ms
from mindone.diffusers import UnCLIPScheduler, DDPMScheduler, StableUnCLIPPipeline
from mindone.diffusers.models import PriorTransformer
from mindone.transformers import CLIPTextModelWithProjection
from transformers import CLIPTokenizer

prior_model_id = "kakaobrain/karlo-v1-alpha"
data_type = ms.float16
prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", mindspore_dtype=data_type)

prior_text_model_id = "openai/clip-vit-large-patch14"
prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, mindspore_dtype=data_type)
prior_scheduler = UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)

stable_unclip_model_id = "stabilityai/stable-diffusion-2-1-unclip-small"

pipe = StableUnCLIPPipeline.from_pretrained(
    stable_unclip_model_id,
    mindspore_dtype=data_type,
    prior_tokenizer=prior_tokenizer,
    prior_text_encoder=prior_text_model,
    prior=prior,
    prior_scheduler=prior_scheduler,
)

wave_prompt = "dramatic wave, the Oceans roar, Strong wave spiral across the oceans as the waves unfurl into roaring crests; perfect wave form; perfect wave shape; dramatic wave shape; wave shape unbelievable; wave; wave shape spectacular"

image = pipe(prompt=wave_prompt)[0][0]
image
```

!!! warning

	For text-to-image we use `stabilityai/stable-diffusion-2-1-unclip-small` as it was trained on CLIP ViT-L/14 embedding, the same as the Karlo model prior. [stabilityai/stable-diffusion-2-1-unclip](https://hf.co/stabilityai/stable-diffusion-2-1-unclip) was trained on OpenCLIP ViT-H, so we don't recommend its use.



### Text guided Image-to-Image Variation

```python
from mindone.diffusers import StableUnCLIPImg2ImgPipeline
from mindone.diffusers.utils import load_image
import mindspore as ms

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", mindspore_dtype=ms.float16, variant="fp16"
)

url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/tarsila_do_amaral.png"
init_image = load_image(url)

images = pipe(init_image)[0]
images[0].save("variation_image.png")
```

Optionally, you can also pass a prompt to `pipe` such as:

```python
prompt = "A fantasy landscape, trending on artstation"

image = pipe(init_image, prompt=prompt)[0][0]
image
```

!!! tip

	Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality.

::: mindone.diffusers.StableUnCLIPPipeline

::: mindone.diffusers.StableUnCLIPImg2ImgPipeline

::: mindone.diffusers.pipelines.ImagePipelineOutput
