<!--Copyright 2024 The HuggingFace Team. All rights reserved.
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

# CogVideoX

[CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) from Tsinghua University & ZhipuAI, by Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, Jie Tang.

The abstract from the paper is:

*We introduce CogVideoX, a large-scale diffusion transformer model designed for generating videos based on text prompts. To efficently model video data, we propose to levearge a 3D Variational Autoencoder (VAE) to compresses videos along both spatial and temporal dimensions. To improve the text-video alignment, we propose an expert transformer with the expert adaptive LayerNorm to facilitate the deep fusion between the two modalities. By employing a progressive training technique, CogVideoX is adept at producing coherent, long-duration videos characterized by significant motion. In addition, we develop an effectively text-video data processing pipeline that includes various data preprocessing strategies and a video captioning method. It significantly helps enhance the performance of CogVideoX, improving both generation quality and semantic alignment. Results show that CogVideoX demonstrates state-of-the-art performance across both multiple machine metrics and human evaluations. The model weight of CogVideoX-2B is publicly available at https://github.com/THUDM/CogVideo.*

!!! tip

    Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.


This pipeline was contributed by [zRzRzRzRzRzRzR](https://github.com/zRzRzRzRzRzRzR). The original codebase can be found [here](https://huggingface.co/THUDM). The original weights can be found under [hf.co/THUDM](https://huggingface.co/THUDM).

There are two models available that can be used with the text-to-video and video-to-video CogVideoX pipelines:
- [`THUDM/CogVideoX-2b`](https://huggingface.co/THUDM/CogVideoX-2b): The recommended dtype for running this model is `fp16`.
- [`THUDM/CogVideoX-5b`](https://huggingface.co/THUDM/CogVideoX-5b): The recommended dtype for running this model is `bf16`.

There is one model available that can be used with the image-to-video CogVideoX pipeline:
- [`THUDM/CogVideoX-5b-I2V`](https://huggingface.co/THUDM/CogVideoX-5b-I2V): The recommended dtype for running this model is `bf16`.

## Inference

First, load the pipeline:

```python
import mindspore
from mindone.diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
from mindone.diffusers.utils import export_to_video,load_image
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b") # or "THUDM/CogVideoX-2b"
```

If you are using the image-to-video pipeline, load it as follows:

```python
pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V")
```

Run inference:

```python
# CogVideoX works well with long and well-described prompts
prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50)[0][0]
```

### Memory optimization

CogVideoX-2b requires about 19 GB of device memory to decode 49 frames (6 seconds of video at 8 FPS) with output resolution 720x480 (W x H), which makes it not possible to run on consumer devices or free-tier T4 Colab. The following memory optimizations could be used to reduce the memory footprint.

- `pipe.vae.enable_tiling()`:
- `pipe.vae.enable_slicing()`


::: mindone.diffusers.CogVideoXPipeline

::: mindone.diffusers.CogVideoXImageToVideoPipeline

::: mindone.diffusers.CogVideoXVideoToVideoPipeline

::: mindone.diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput
