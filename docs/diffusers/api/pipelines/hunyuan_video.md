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

# HunyuanVideo

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[HunyuanVideo](https://www.arxiv.org/abs/2412.03603) by Tencent.

*Recent advancements in video generation have significantly impacted daily life for both individuals and industries. However, the leading video generation models remain closed-source, resulting in a notable performance gap between industry capabilities and those available to the public. In this report, we introduce HunyuanVideo, an innovative open-source video foundation model that demonstrates performance in video generation comparable to, or even surpassing, that of leading closed-source models. HunyuanVideo encompasses a comprehensive framework that integrates several key elements, including data curation, advanced architectural design, progressive model scaling and training, and an efficient infrastructure tailored for large-scale model training and inference. As a result, we successfully trained a video generative model with over 13 billion parameters, making it the largest among all open-source models. We conducted extensive experiments and implemented a series of targeted designs to ensure high visual quality, motion dynamics, text-video alignment, and advanced filming techniques. According to evaluations by professionals, HunyuanVideo outperforms previous state-of-the-art models, including Runway Gen-3, Luma 1.6, and three top-performing Chinese video generative models. By releasing the code for the foundation model and its applications, we aim to bridge the gap between closed-source and open-source communities. This initiative will empower individuals within the community to experiment with their ideas, fostering a more dynamic and vibrant video generation ecosystem. The code is publicly available at [this https URL](https://github.com/tencent/HunyuanVideo).*

!!! tip

    Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

Recommendations for inference:
- Both text encoders should be in `ms.float16`.
- Transformer should be in `ms.bfloat16`.
- VAE should be in `ms.float16`.
- `num_frames` should be of the form `4 * k + 1`, for example `49` or `129`.
- For smaller resolution videos, try lower values of `shift` (between `2.0` to `5.0`) in the [Scheduler](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/flow_match_euler_discrete/#mindone.diffusers.FlowMatchEulerDiscreteScheduler). For larger resolution images, try higher values (between `7.0` and `12.0`). The default value is `7.0` for HunyuanVideo.
- For more information about supported resolutions and other details, please refer to the original repository [here](https://github.com/Tencent/HunyuanVideo/).

## Available models

The following models are available for the [`HunyuanVideoPipeline`](text-to-video) pipeline:

| Model name | Description |
|:---|:---|
| [`hunyuanvideo-community/HunyuanVideo`](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) | Official HunyuanVideo (guidance-distilled). Performs best at multiple resolutions and frames. Performs best with `guidance_scale=6.0`, `true_cfg_scale=1.0` and without a negative prompt. |
| [`https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-T2V`](https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-T2V) | Skywork's custom finetune of HunyuanVideo (de-distilled). Performs best with `97x544x960` resolution, `guidance_scale=1.0`, `true_cfg_scale=6.0` and a negative prompt. |

The following models are available for the image-to-video pipeline:

| Model name | Description |
|:---|:---|
| [`Skywork/SkyReels-V1-Hunyuan-I2V`](https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-I2V) | Skywork's custom finetune of HunyuanVideo (de-distilled). Performs best with `97x544x960` resolution. Performs best at `97x544x960` resolution, `guidance_scale=1.0`, `true_cfg_scale=6.0` and a negative prompt. |
| [`hunyuanvideo-community/HunyuanVideo-I2V-33ch`](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-I2V) | Tecent's official HunyuanVideo 33-channel I2V model. Performs best at resolutions of 480, 720, 960, 1280. A higher `shift` value when initializing the scheduler is recommended (good values are between 7 and 20). |
| [`hunyuanvideo-community/HunyuanVideo-I2V`](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-I2V) | Tecent's official HunyuanVideo 16-channel I2V model. Performs best at resolutions of 480, 720, 960, 1280. A higher `shift` value when initializing the scheduler is recommended (good values are between 7 and 20) |

::: mindone.diffusers.HunyuanVideoPipeline

::: mindone.diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput
