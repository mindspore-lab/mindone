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
# limitations under the License.
-->

# Mochi 1 Preview

[Mochi 1 Preview](https://huggingface.co/genmo/mochi-1-preview) from Genmo.

*Mochi 1 preview is an open state-of-the-art video generation model with high-fidelity motion and strong prompt adherence in preliminary evaluation. This model dramatically closes the gap between closed and open video generation systems. The model is released under a permissive Apache 2.0 license.*

!!! tip

    Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.


## Generating videos with Mochi-1 Preview

The following example will download the full precision `mochi-1-preview` weights and produce the highest quality results but will require at least 42GB VRAM to run.

```python
import mindspore as ms
from mindone.diffusers import MochiPipeline
from mindone.diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", mindspore_dtype=ms.float16)

# Enable memory savings
pipe.enable_vae_tiling()

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."

frames = pipe(prompt, num_inference_steps=28, guidance_scale=3.5)[0][0]

export_to_video(frames, "mochi.mp4", fps=30)
```

## Using a lower precision variant to save memory

The following example will use the `bfloat16` variant of the model and requires 22GB VRAM to run. There is a slight drop in the quality of the generated video as a result.

```python
import mindspore as ms
from mindone.diffusers import MochiPipeline
from mindone.diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", mindspore_dtype=ms.bfloat16)

# Enable memory savings
pipe.enable_vae_tiling()

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
frames = pipe(prompt, num_frames=85)[0][0]

export_to_video(frames, "mochi.mp4", fps=30)
```

## Using single file loading with the Mochi Transformer

You can use `from_single_file` to load the Mochi transformer in its original format.

!!! tip
    Diffusers currently doesn't support using the FP8 scaled versions of the Mochi single file checkpoints.

```python
import mindspore as ms
from mindone.diffusers import MochiPipeline, MochiTransformer3DModel
from mindone.diffusers.utils import export_to_video

model_id = "genmo/mochi-1-preview"

ckpt_path = "https://huggingface.co/Comfy-Org/mochi_preview_repackaged/blob/main/split_files/diffusion_models/mochi_preview_bf16.safetensors"

transformer = MochiTransformer3DModel.from_pretrained(ckpt_path, mindspore_dtype=ms.bfloat16)

pipe = MochiPipeline.from_pretrained(model_id,  transformer=transformer)
pipe.enable_vae_tiling()

frames = pipe(
    prompt="Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k.",
    negative_prompt="",
    height=480,
    width=848,
    num_frames=85,
    num_inference_steps=50,
    guidance_scale=4.5,
    num_videos_per_prompt=1,
    generator=torch.Generator(device="cuda").manual_seed(0),
    max_sequence_length=256,
    output_type="pil",
)[0][0]

export_to_video(frames, "output.mp4", fps=30)
```

::: mindone.diffusers.MochiPipeline

::: mindone.diffusers.pipelines.mochi.pipeline_output.MochiPipelineOutput
