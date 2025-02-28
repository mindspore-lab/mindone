<div align="center">
<h1>🚀 Wan: Open and Advanced Large-Scale Video Generative Models </h1>

</div>

In this repository, we present an efficient MindSpore implementation of [Wan2.1](https://github.com/Wan-Video/Wan2.1). This repository is built on the models and code released by the Alibaba Wan group. We are grateful for their exceptional work and generous contribution to open source.

- 👍 **SOTA Performance**: **Wan2.1** consistently outperforms existing open-source models and state-of-the-art commercial solutions across multiple benchmarks.
- 👍 **Multiple Tasks**: **Wan2.1** excels in Text-to-Video, Image-to-Video, Video Editing, Text-to-Image, and Video-to-Audio, advancing the field of video generation.
- 👍 **Visual Text Generation**: **Wan2.1** is the first video model capable of generating both Chinese and English text, featuring robust text generation that enhances its practical applications.
- 👍 **Powerful Video VAE**: **Wan-VAE** delivers exceptional efficiency and performance, encoding and decoding 1080P videos of any length while preserving temporal information, making it an ideal foundation for video and image generation.


## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/4aca6063-60bf-4953-bfb7-e265053f49ef" width="70%" poster=""> </video>
</div>

## 🔥 Latest News!!

* Feb 28, 2025: 👋 MindSpore implementation of Wan2.1 is released, supporting text-to-video and image-to-video inference tasks on 1.3B and 14B  models.


## 📑 Todo List
- Wan2.1 Text-to-Video
    - [x] Single-NPU Inference code of the 14B and 1.3B models
    - [x] Multi-NPU Inference code of the 14B models
    - [ ] Gradio demo
- Wan2.1 Image-to-Video
    - [x] Single-NPU Inference code of the 14B model
    - [x] Multi-NPU Inference code of the 14B model
    - [ ] Gradio demo


## Quickstart

###  Requirments

The code is tested in the following environments

| mindspore | ascend driver | firmware | cann tookit/kernel |
| :---:     |   :---:       | :---:    | :---:              |
| 2.5.0     |  24.1.0     |7.35.23    |   8.0.RC3.beta1   |


### Installation
Clone the repo:
```
git clone https://github.com/mindspore-lab/mindone
cd mindone/examples/wan2_1
```

Install dependencies:
```
pip install -r requirements.txt
```

### Model Download

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| T2V-14B       |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)      🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B)          | Supports both 480P and 720P
| I2V-14B-720P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)     | Supports 720P
| I2V-14B-480P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)      | Supports 480P
| T2V-1.3B      |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)     🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)         | Supports 480P

> 💡Note: The 1.3B model is capable of generating videos at 720P resolution. However, due to limited training at this resolution, the results are generally less stable compared to 480P. For optimal performance, we recommend using 480P resolution.


Download the models using huggingface-cli or modelscope:
```
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P
```

Download models using modelscope-cli similarly:
```
pip install modelscope
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
```

### Run Text-to-Video Generation

This repository supports two Text-to-Video models (1.3B and 14B) and two resolutions (480P and 720P). The parameters and configurations for these models are as follows:

<table>
    <thead>
        <tr>
            <th rowspan="2">Task</th>
            <th colspan="2">Resolution</th>
            <th rowspan="2">Model</th>
        </tr>
        <tr>
            <th>480P</th>
            <th>720P</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>t2v-14B</td>
            <td style="color: green;">✔️</td>
            <td style="color: green;">✔️</td>
            <td>Wan2.1-T2V-14B</td>
        </tr>
        <tr>
            <td>t2v-1.3B</td>
            <td style="color: green;">✔️</td>
            <td style="color: red;">❌</td>
            <td>Wan2.1-T2V-1.3B</td>
        </tr>
    </tbody>
</table>


- Single-NPU inference

```
python generate.py  \
    --task t2v-14B \
    --size 1280*720 \
    --ckpt_dir ./Wan2.1-T2V-14B \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

```
python generate.py  \
    --task t2v-1.3B \
    --size 832*480 \
    --ckpt_dir ./Wan2.1-T2V-1.3B \
    --sample_shift 8 \
    --sample_guide_scale 6 \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

> 💡Note: If you are using the `T2V-1.3B` model, we recommend setting the parameter `--sample_guide_scale 6`. The `--sample_shift parameter` can be adjusted within the range of 8 to 12 based on the performance.


- Multi-NPU inference

```
worker_num=2
msrun --worker_num=${worker_num} generate.py \
    --task t2v-14B \
    --size 1280*720 \
    --ckpt_dir ./Wan2.1-T2V-14B \
    --dit_fsdp --t5_fsdp \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```


### Run Image-to-Video Generation

Similar to Text-to-Video, Image-to-Video supports different resolutions. The specific parameters and their corresponding settings are as follows:

<table>
    <thead>
        <tr>
            <th rowspan="2">Task</th>
            <th colspan="2">Resolution</th>
            <th rowspan="2">Model</th>
        </tr>
        <tr>
            <th>480P</th>
            <th>720P</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>i2v-14B</td>
            <td style="color: green;">❌</td>
            <td style="color: green;">✔️</td>
            <td>Wan2.1-I2V-14B-720P</td>
        </tr>
        <tr>
            <td>i2v-14B</td>
            <td style="color: green;">✔️</td>
            <td style="color: red;">❌</td>
            <td>Wan2.1-T2V-14B-480P</td>
        </tr>
    </tbody>
</table>


- Single-NPU inference
```
python generate.py \
    --task i2v-14B \
    --size 832*480 \
    --ckpt_dir ./Wan2.1-I2V-14B-480P \
    --image examples/i2v_input.JPG \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```
> I2V 720P requires multi-NPU inference

> 💡For the Image-to-Video task, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.


- Multi-NPU inference

```
worker_num=2
msrun --nproc_per_node=${worker_num} generate.py \
    --task i2v-14B --size 1280*720 \
    --ckpt_dir ./Wan2.1-I2V-14B-720P \
    --dit_fsdp --t5_fsdp \
    --image examples/i2v_input.JPG \
    --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```


## Citation

```
@article{wan2.1,
    title   = {Wan: Open and Advanced Large-Scale Video Generative Models},
    author  = {Wan Team},
    journal = {},
    year    = {2025}
}
```

## License Agreement
The models in this repository are licensed under the Apache 2.0 License. We claim no rights over the your generate contents, granting you the freedom to use them while ensuring that your usage complies with the provisions of this license. You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws, causes harm to individuals or groups, disseminates personal information intended for harm, spreads misinformation, or targets vulnerable populations. For a complete list of restrictions and details regarding your rights, please refer to the full text of the [license](LICENSE.txt).


## Acknowledgements

We would like to thank the contributors to the [Wan2.1](https://github.com/Wan-Video/Wan2.1), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research.

