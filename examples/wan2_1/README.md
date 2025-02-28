<div align="center">
<h1>🚀 Wan: Open and Advanced Large-Scale Video Generative Models </h1>

</div>

In this repository, we present an efficient MindSpore implementation of [Wan2.1](https://github.com/Wan-Video/Wan2.1). This repository is built on the models and code released by the Alibaba Wan group. We are grateful for their exceptional work and generous contribution to open source.

- 👍 **SOTA Performance**: **Wan2.1** consistently outperforms existing open-source models and state-of-the-art commercial solutions across multiple benchmarks.
- 👍 **Multiple Tasks**: **Wan2.1** excels in Text-to-Video, Image-to-Video, Video Editing, Text-to-Image, and Video-to-Audio, advancing the field of video generation.
- 👍 **Visual Text Generation**: **Wan2.1** is the first video model capable of generating both Chinese and English text, featuring robust text generation that enhances its practical applications.
- 👍 **Powerful Video VAE**: **Wan-VAE** delivers exceptional efficiency and performance, encoding and decoding 1080P videos of any length while preserving temporal information, making it an ideal foundation for video and image generation.


## Video Demos

The following videos are generated based on MindSpore and Ascend 910*.

- Text-to-Video

https://github.com/user-attachments/assets/f6705d28-7755-447b-a256-6727f66d693b

```text
prompt: A sepia-toned vintage photograph depicting a whimsical bicycle race featuring several dogs wearing goggles and tiny cycling outfits. The canine racers, with determined expressions and blurred motion, pedal miniature bicycles on a dusty road. Spectators in period clothing line the sides, adding to the nostalgic atmosphere. Slightly grainy and blurred, mimicking old photos, with soft side lighting enhancing the warm tones and rustic charm of the scene. 'Bicycle Race' captures this unique moment in a medium shot, focusing on both the racers and the lively crowd.
```

https://github.com/user-attachments/assets/1e1da53a-9112-4fc3-bb8e-b458497c4806

```text
prompt: Film quality, professional quality, rich details. The video begins to show the surface of a pond, and the camera slowly zooms in to a close-up. The water surface begins to bubble, and then a blonde woman is seen coming out of the lotus pond soaked all over, showing the subtle changes in her facial expression, creating a dreamy atmosphere.
```

https://github.com/user-attachments/assets/34e4501f-a207-40bb-bb6c-b162ff6505b0

```text
prompt: Two anthropomorphic cats wearing boxing suits and bright gloves fiercely battled on the boxing ring under the spotlight. Their muscles are tight, displaying the strength and agility of professional boxers. A spotted dog judge stood aside. The animals in the audience around cheered and cheered, adding a lively atmosphere to the competition. The cat's boxing movements are quick and powerful, with its paws tracing blurry trajectories in the air. The screen adopts a dynamic blur effect, close ups, and focuses on the intense confrontation on the boxing ring.
```

https://github.com/user-attachments/assets/aceda253-78a2-4fa5-9edc-83f035c7c2ea

```text
prompt: Sports photography full of dynamism, several motorcycles fiercely compete on the loess flying track, their wheels rolling up the dust in the sky. The motorcyclist is wearing professional racing clothes. The camera uses a high-speed shutter to capture moments, follows from the side and rear, and finally freezes in a close-up of a motorcycle, showcasing its exquisite body lines and powerful mechanical beauty, creating a tense and exciting racing atmosphere. Close up dynamic perspective, perfectly presenting the visual impact of speed and power.
```

- Image-to-Video


https://github.com/user-attachments/assets/d37bf480-595e-4a41-95f8-acbc421b7428

```text
prompt: Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.
```


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
    --dit_fsdp --t5_fsdp --ulysses_sp \
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

> 💡For the Image-to-Video task, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.


- Multi-NPU inference

```
msrun --work_num=2 generate.py \
    --task i2v-14B --size 1280*720 \
    --ckpt_dir ./Wan2.1-I2V-14B-720P \
    --dit_fsdp --t5_fsdp --ulysses_sp \
    --image examples/i2v_input.JPG \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

 > 💡At least 2 cards are required to run 720P I2V generation to avoid OOM. 8 cards will accelerate the generation process at most.


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

