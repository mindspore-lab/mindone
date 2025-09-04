# Wan2.2 (MindSpore)

This repository provides the inference codes of [Wan2.2](https://arxiv.org/abs/2503.20314), adapted from [offical Wan2.2](https://github.com/Wan-Video/Wan2.2) to support MindSpore.

-----

[**Wan: Open and Advanced Large-Scale Video Generative Models**](https://arxiv.org/abs/2503.20314) <be>


We are excited to introduce **Wan2.2**, a major upgrade to our foundational video models. With **Wan2.2**, we have focused on incorporating the following innovations:

- 👍 **Effective MoE Architecture**: Wan2.2 introduces a Mixture-of-Experts (MoE) architecture into video diffusion models. By separating the denoising process cross timesteps with specialized powerful expert models, this enlarges the overall model capacity while maintaining the same computational cost.

- 👍 **Cinematic-level Aesthetics**: Wan2.2 incorporates meticulously curated aesthetic data, complete with detailed labels for lighting, composition, contrast, color tone, and more. This allows for more precise and controllable cinematic style generation, facilitating the creation of videos with customizable aesthetic preferences.

- 👍 **Complex Motion Generation**: Compared to Wan2.1, Wan2.2 is trained on a significantly larger data, with +65.6% more images and +83.2% more videos. This expansion notably enhances the model's generalization across multiple dimensions such as motions,  semantics, and aesthetics, achieving TOP performance among all open-sourced and closed-sourced models.

- 👍 **Efficient High-Definition Hybrid TI2V**:  Wan2.2 open-sources a 5B model built with our advanced Wan2.2-VAE that achieves a compression ratio of **16×16×4**. This model supports both text-to-video and image-to-video generation at 720P resolution with 24fps and can also run on consumer-grade graphics cards like 4090. It is one of the fastest **720P@24fps** models currently available, capable of serving both the industrial and academic sectors simultaneously.


## 📑 Todo List
- Wan2.2 Text-to-Video
    - [x] Multi-NPU Inference code of the A14B and 14B models
- Wan2.2 Image-to-Video
    - [x] Multi-NPU Inference code of the A14B model
- Wan2.2 Text-Image-to-Video
    - [x] Single-NPU/Multi-NPU Inference code of the 5B model
- Wan2.2-S2V Speech-to-Video
    - [x] Multi-NPU Inference code of Wan2.2-S2V

## Run Wan2.2

#### Requirements
|mindspore |	ascend driver | firmware | cann tookit/kernel|
|--- | --- | --- | --- |
|2.7.0 | 24.1RC3 | 7.3.0.1.231 | 8.2.RC1 |


#### Installation
Clone the repo:
```sh
git clone https://github.com/mindspore-lab/mindone.git
cd mindone/examples/wan2_2
```

Install dependencies:
```sh
pip install -r requirements.txt
```


#### Model Download

| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |
| S2V-14B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video model, supports 480P & 720P |


> 💡Note:
> The TI2V-5B model supports 720P video generation at **24 FPS**.


Download models using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

Download models using modelscope-cli:
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

#### Run Text-to-Video Generation

This repository supports the `Wan2.2-T2V-A14B` Text-to-Video model and can simultaneously support video generation at 480P and 720P resolutions.


##### (1) Without Prompt Extension

To facilitate implementation, we will start with a basic version of the inference process that skips the [prompt extension](#2-using-prompt-extention) step.

- Multi-NPU inference using ZeRO3 + DeepSpeed Ulysses

  We use [ZeRO3](https://arxiv.org/abs/1910.02054) and [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509) to accelerate inference.

``` sh
msrun --worker_num=4 --local_worker_num=4 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_zero3 --t5_zero3 --ulysses_size 4 --offload_model True --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

> 💡To reduce NPU memory usage, you can use the `--convert_model_dtype` options.


##### (2) Using Prompt Extension

Extending the prompts can effectively enrich the details in the generated videos, further enhancing the video quality. Therefore, we recommend enabling prompt extension. We provide the following two methods for prompt extension:

- Using a local model for extension.

  - By default, the Qwen model on HuggingFace is used for this extension. Users can choose Qwen models or other models based on the available NPU memory size.
  - For text-to-video tasks, you can use models like `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`.
  - For image-to-video tasks, you can use models like `Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen2.5-VL-3B-Instruct`.
  - Larger models generally provide better extension results but require more NPU memory.
  - You can modify the model used for extension with the parameter `--prompt_extend_model` , allowing you to specify either a local model path or a Hugging Face model. For example:

``` sh
msrun --worker_num=4 --local_worker_num=4 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_zero3 --t5_zero3 --local_qwen_zero3 --ulysses_size 4 --offload_model True --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
```


#### Run Image-to-Video Generation

This repository supports the `Wan2.2-I2V-A14B` Image-to-Video model and can simultaneously support video generation at 480P and 720P resolutions.


- Multi-NPU inference using ZeRO3 + DeepSpeed Ulysses

```sh
msrun --worker_num=4 --local_worker_num=4 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --dit_zero3 --t5_zero3 --ulysses_size 4 --offload_model True --image ../wan2_1/examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> 💡For the Image-to-Video task, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.


- Image-to-Video Generation without prompt

```sh
msrun --worker_num=4 --local_worker_num=4 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image ../wan2_1/examples/i2v_input.JPG --dit_zero3 --t5_zero3 --local_qwen_zero3 --ulysses_size 4 --offload_model True --use_prompt_extend --prompt_extend_method 'local_qwen'
```

> 💡The model can generate videos solely from the input image. You can use prompt extension to generate prompt from the image.


#### Run Text-Image-to-Video Generation

This repository supports the `Wan2.2-TI2V-5B` Text-Image-to-Video model and can support video generation at 720P resolutions.


- Single-NPU Text-to-Video inference
```sh
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

> 💡Unlike other tasks, the 720P resolution of the Text-Image-to-Video task is `1280*704` or `704*1280`.


- Single-NPU Image-to-Video inference
```sh
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --convert_model_dtype --image ../wan2_1/examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> 💡If the image parameter is configured, it is an Image-to-Video generation; otherwise, it defaults to a Text-to-Video generation.

> 💡Similar to Image-to-Video, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.


- Multi-NPU inference using ZeRO3 + DeepSpeed Ulysses

```sh
msrun --worker_num=4 --local_worker_num=4 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_zero3 --t5_zero3 --ulysses_size 4 --image ../wan2_1/examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```


#### Run Speech-to-Video Generation

This repository supports the `Wan2.2-S2V-14B` Speech-to-Video model and can simultaneously support video generation at 480P and 720P resolutions.

- Multi-GPU inference using ZeRO3 + DeepSpeed Ulysses

```sh
msrun --worker_num=4 --local_worker_num=4 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_zero3 --t5_zero3 --ulysses_size 4 --offload_model True --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." --image "../wan2_1/examples/i2v_input.JPG" --audio "examples/talk.wav"
```

> You can get the `talk.wav` file from [official Wan2.2](https://github.com/Wan-Video/Wan2.2).

- Pose + Audio driven generation

```sh
msrun --worker_num=4 --local_worker_num=4 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_zero3 --t5_zero3 --ulysses_size 4 --offload_model True --prompt "a person is singing" --image "examples/pose.png" --audio "examples/sing.MP3" --pose_video "examples/pose.mp4"
```

> 💡For the Speech-to-Video task, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.

> 💡The model can generate videos from audio input combined with reference image and optional text prompt.

> 💡The `--pose_video` parameter enables pose-driven generation, allowing the model to follow specific pose sequences while generating videos synchronized with audio input.

> 💡The `--num_clip` parameter controls the number of video clips generated, useful for quick preview with shorter generation time.

> You can get the `pose.png`, `sing.MP3` and `pose.mp4` files from [official Wan2.2](https://github.com/Wan-Video/Wan2.2).


## Citation
If you find this work helpful, please cite.

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## Acknowledgements

We would like to thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research.
