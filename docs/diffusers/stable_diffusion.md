<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Basic performance

Diffusion is a random process that is computationally demanding. You may need to run the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) several times before getting a desired output. That's why it's important to carefully balance generation speed and memory usage in order to iterate faster,

This guide recommends some basic performance tips for using the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline). Refer to the Inference Optimization section docs such as [Accelerate inference](https://mindspore-lab.github.io/mindone/latest/diffusers/optimization/fp16) or [Reduce memory usage](https://mindspore-lab.github.io/mindone/latest/diffusers/optimization/memory) for more detailed performance guides.


## Inference speed

Denoising is the most computationally demanding process during diffusion. Methods that optimizes this process accelerates inference speed. Try the following methods for a speed up.

- One of the simplest ways to speed up inference is to place the pipeline on a NPU the same way you would with any Mindspore cell.
That is, do nothing! MindSpore will automatically take care of model placement, so you don't need to:

```diff
- pipeline = pipeline.to("cuda")
```

- Set `mindspore_dtype=mindspore.bfloat16` to execute the pipeline in half-precision. Reducing the data type precision increases speed because it takes less time to perform computations in a lower precision.

```py
import mindspore as ms
import time
from mindone.diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  mindspore_dtype=ms.bfloat16,
)
```

- Use a faster scheduler, such as [`DPMSolverMultistepScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/multistep_dpm_solver/#mindone.diffusers.DPMSolverMultistepScheduler), which only requires ~20-25 steps.
- Set `num_inference_steps` to a lower value. Reducing the number of inference steps reduces the overall number of computations. However, this can result in lower generation quality.

```py
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

start_time = time.perf_counter()
image = pipeline(prompt)[0][0]
end_time = time.perf_counter()

print(f"Image generation took {end_time - start_time:.3f} seconds")
```

## Generation quality

Many modern diffusion models deliver high-quality images out-of-the-box. However, you can still improve generation quality by trying the following.

- Try a more detailed and descriptive prompt. Include details such as the image medium, subject, style, and aesthetic. A negative prompt may also help by guiding a model away from undesirable features by using words like low quality or blurry.

    ```py
    import mindspore as ms
    from mindone.diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        mindspore_dtype=ms.bfloat16,
    )

    prompt = """
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
    """
    negative_prompt = "low quality, blurry, ugly, poor details"
    pipeline(prompt, negative_prompt=negative_prompt)[0][0]
    ```

    For more details about creating better prompts, take a look at the [Prompt techniques](./using-diffusers/weighted_prompts) doc.

- Try a different scheduler, like [`HeunDiscreteScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/heun/#mindone.diffusers.HeunDiscreteScheduler) or [`LMSDiscreteScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/lms_discrete/#mindone.diffusers.LMSDiscreteScheduler), that gives up generation speed for quality.

    ```py
    import mindspore as ms
    from mindone.diffusers import DiffusionPipeline, HeunDiscreteScheduler

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        mindspore_dtype=ms.bfloat16,
    )
    pipeline.scheduler = HeunDiscreteScheduler.from_config(pipeline.scheduler.config)

    prompt = """
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
    """
    negative_prompt = "low quality, blurry, ugly, poor details"
    pipeline(prompt, negative_prompt=negative_prompt)[0][0]
    ```
