<!-- Copyright 2024 The HuggingFace Team. All rights reserved.
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

# LTX Video

[LTX Video](https://huggingface.co/Lightricks/LTX-Video) is the first DiT-based video generation model capable of generating high-quality videos in real-time. It produces 24 FPS videos at a 768x512 resolution faster than they can be watched. Trained on a large-scale dataset of diverse videos, the model generates high-resolution videos with realistic and varied content. We provide a model for both text-to-video as well as image + text-to-video usecases.

!!! tip

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality.

Available models:

|                                                 Model name                                                 | Recommended dtype |
|:----------------------------------------------------------------------------------------------------------:|:-----------------:|
| [`LTX Video 0.9.0`](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.safetensors)   |   `ms.bfloat16`   |
| [`LTX Video 0.9.1`](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors) |   `ms.bfloat16`   |

Note: The recommended dtype is for the transformer component. The VAE and text encoders can be either `ms.float32`, `ms.bfloat16` or `ms.float16` but the recommended dtype is `ms.bfloat16` as used in the original repository.

## Loading Single Files

Loading the original LTX Video checkpoints is also possible with [`~ModelMixin.from_single_file`]. We recommend using `from_single_file` for the Lightricks series of models, as they plan to release multiple models in the future in the single file format.

```python
import mindspore as ms
from mindone.diffusers import AutoencoderKLLTXVideo, LTXImageToVideoPipeline, LTXVideoTransformer3DModel

# `single_file_url` could also be https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.1.safetensors
single_file_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.safetensors"
transformer = LTXVideoTransformer3DModel.from_single_file(
  single_file_url, mindspore_dtype=ms.bfloat16
)
vae = AutoencoderKLLTXVideo.from_single_file(single_file_url, mindspore_dtype=ms.bfloat16)
pipe = LTXImageToVideoPipeline.from_pretrained(
  "Lightricks/LTX-Video", transformer=transformer, vae=vae, mindspore_dtype=ms.bfloat16
)

# ... inference code ...
```

Alternatively, the pipeline can be used to load the weights with [`~FromSingleFileMixin.from_single_file`].

```python
import mindspore as ms
from mindone.diffusers import LTXImageToVideoPipeline
from mindone.transformers import T5EncoderModel
from transformers import T5Tokenizer

single_file_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.safetensors"
text_encoder = T5EncoderModel.from_pretrained(
  "Lightricks/LTX-Video", subfolder="text_encoder", mindspore_dtype=ms.bfloat16
)
tokenizer = T5Tokenizer.from_pretrained(
  "Lightricks/LTX-Video", subfolder="tokenizer", mindspore_dtype=ms.bfloat16
)
pipe = LTXImageToVideoPipeline.from_single_file(
  single_file_url, text_encoder=text_encoder, tokenizer=tokenizer, mindspore_dtype=ms.bfloat16
)
```

Loading and running inference with [LTX Video 0.9.1](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors) weights.

```python
import mindspore as ms
from mindone.diffusers import LTXPipeline
from mindone.diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", mindspore_dtype=ms.bfloat16)

prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
)[0][0]
export_to_video(video, "output.mp4", fps=24)
```

Refer to [this section](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/cogvideox/#memory-optimization) to learn more about optimizing memory consumption.

::: mindone.diffusers.LTXPipeline

::: mindone.diffusers.LTXImageToVideoPipeline

::: mindone.diffusers.pipelines.ltx.pipeline_output.LTXPipelineOutput
