<!--Copyright 2025 The HuggingFace Team. All rights reserved.
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

# EasyAnimate
[EasyAnimate](https://github.com/aigc-apps/EasyAnimate) by Alibaba PAI.

The description from it's GitHub page:
*EasyAnimate is a pipeline based on the transformer architecture, designed for generating AI images and videos, and for training baseline models and Lora models for Diffusion Transformer. We support direct prediction from pre-trained EasyAnimate models, allowing for the generation of videos with various resolutions, approximately 6 seconds in length, at 8fps (EasyAnimateV5.1, 1 to 49 frames). Additionally, users can train their own baseline and Lora models for specific style transformations.*

This pipeline was contributed by [bubbliiiing](https://github.com/bubbliiiing). The original codebase can be found [here](https://huggingface.co/alibaba-pai). The original weights can be found under [hf.co/alibaba-pai](https://huggingface.co/alibaba-pai).

There are two official EasyAnimate checkpoints for text-to-video and video-to-video.

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`alibaba-pai/EasyAnimateV5.1-12b-zh`](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh) | mindspore.float16 |
| [`alibaba-pai/EasyAnimateV5.1-12b-zh-InP`](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-InP) | mindspore.float16 |

There is one official EasyAnimate checkpoints available for image-to-video and video-to-video.

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`alibaba-pai/EasyAnimateV5.1-12b-zh-InP`](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-InP) | mindspore.float16 |

There are two official EasyAnimate checkpoints available for control-to-video.

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`alibaba-pai/EasyAnimateV5.1-12b-zh-Control`](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control) | mindspore.float16 |
| [`alibaba-pai/EasyAnimateV5.1-12b-zh-Control-Camera`](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control-Camera) | mindspore.float16 |

For the EasyAnimateV5.1 series:
- Text-to-video (T2V) and Image-to-video (I2V) works for multiple resolutions. The width and height can vary from 256 to 1024.
- Both T2V and I2V models support generation with 1~49 frames and work best at this value. Exporting videos at 8 FPS is recommended.

::: mindone.diffusers.EasyAnimatePipeline

::: mindone.diffusers.pipelines.easyanimate.pipeline_output.EasyAnimatePipelineOutput
