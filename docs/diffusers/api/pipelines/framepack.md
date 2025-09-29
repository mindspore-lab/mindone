<!-- Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# Framepack

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[Packing Input Frame Context in Next-Frame Prediction Models for Video Generation](https://huggingface.co/papers/2504.12626) by Lvmin Zhang and Maneesh Agrawala.

*We present a neural network structure, FramePack, to train next-frame (or next-frame-section) prediction models for video generation. The FramePack compresses input frames to make the transformer context length a fixed number regardless of the video length. As a result, we are able to process a large number of frames using video diffusion with computation bottleneck similar to image diffusion. This also makes the training video batch sizes significantly higher (batch sizes become comparable to image diffusion training). We also propose an anti-drifting sampling method that generates frames in inverted temporal order with early-established endpoints to avoid exposure bias (error accumulation over iterations). Finally, we show that existing video diffusion models can be finetuned with FramePack, and their visual quality may be improved because the next-frame prediction supports more balanced diffusion schedulers with less extreme flow shift timesteps.*

!!! tip

    Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## Available models

| Model name                                                                                                    | Description                                                                                                                                                                                             |
|:--------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| - [`lllyasviel/FramePackI2V_HY`](https://huggingface.co/lllyasviel/FramePackI2V_HY)                           | Trained with the "inverted anti-drifting" strategy as described in the paper. Inference requires setting `sampling_type="inverted_anti_drifting"` when running the pipeline.                            |
| - [`lllyasviel/FramePack_F1_I2V_HY_20250503`](https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503) | Trained with a novel anti-drifting strategy but inference is performed in "vanilla" strategy as described in the paper. Inference requires setting `sampling_type="vanilla"` when running the pipeline. |

## Usage

Refer to the pipeline documentation for basic usage examples. The following section contains examples of offloading, different sampling methods, quantization, and more.

### First and last frame to video

The following example shows how to use Framepack with start and end image controls, using the inverted anti-drifiting sampling model.

```python
import mindspore as ms
from mindone.diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
from mindone.diffusers.utils import export_to_video, load_image
from mindone.transformers import SiglipVisionModel
from transformers import SiglipImageProcessor
import numpy as np

transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
    "lllyasviel/FramePackI2V_HY", mindspore_dtype=ms.bfloat16
)
feature_extractor = SiglipImageProcessor.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
)
image_encoder = SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="image_encoder", mindspore_dtype=ms.float16
)
pipe = HunyuanVideoFramepackPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
    mindspore_dtype=ms.float16,
)

# Enable memory optimizations
pipe.vae.enable_tiling()

prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
first_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
)
last_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png"
)
output = pipe(
    image=first_image,
    last_image=last_image,
    prompt=prompt,
    height=512,
    width=512,
    num_frames=91,
    num_inference_steps=30,
    guidance_scale=9.0,
    generator=np.random.Generator(np.random.PCG64(seed=0)),
    sampling_type="inverted_anti_drifting",
)[0][0]
export_to_video(output, "output.mp4", fps=30)
```

### Vanilla sampling

The following example shows how to use Framepack with the F1 model trained with vanilla sampling but new regulation approach for anti-drifting.

```python
import mindspore as ms
from mindone.diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
from mindone.diffusers.utils import export_to_video, load_image
from mindone.transformers import SiglipVisionModel
from transformers import SiglipImageProcessor
import numpy as np

transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
    "lllyasviel/FramePack_F1_I2V_HY_20250503", mindspore_dtype=ms.bfloat16
)
feature_extractor = SiglipImageProcessor.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
)
image_encoder = SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="image_encoder", mindspore_dtype=ms.float16
)
pipe = HunyuanVideoFramepackPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
    mindspore_dtype=ms.float16,
)

# Enable memory optimizations
pipe.vae.enable_tiling()

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
)
output = pipe(
    image=image,
    prompt="A penguin dancing in the snow",
    height=832,
    width=480,
    num_frames=91,
    num_inference_steps=30,
    guidance_scale=9.0,
    generator=np.random.Generator(np.random.PCG64(seed=0)),
    sampling_type="vanilla",
)[0][0]
export_to_video(output, "output.mp4", fps=30)
```

::: mindone.diffusers.HunyuanVideoFramepackPipeline

::: mindone.diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput
