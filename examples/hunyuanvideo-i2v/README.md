# HunyuanVideo-I2V

This is a **MindSpore** implementation of [HunyuanVideo-I2V](https://github.com/Tencent/HunyuanVideo-I2V). It contains the code for **training** and **inference** of HunyuanVideo and 3D CausalVAE.


## 📑 Development Plan

Here is the development plan of the project:

- CausalVAE:
    - [x] Inference
    - [ ] Evalution
    - [ ] Training
- HunyuanVideo (13B):
    - [x] Inference (w. and w.o. LoRA weight)
    - [ ] Training
    - [ ] LoRA finetune



## 📦 Requirements


<div align="center">

| MindSpore | Ascend Driver |  Firmware   | CANN toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
|   2.5.0   |  24.1.RC2     | 7.5.0.2.220 |  8.0.RC3.beta1      |

</div>

1. Install
   [CANN 8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1)
   and MindSpore according to the [official instructions](https://www.mindspore.cn/install).
2. Install requirements
    ```shell
    pip install -r requirements.txt
    ```
## 🎥 Demo

The following videos are generated based on MindSpore and Ascend 910B.

|  Reference Image | Generated Video |
|:----------------:|:----------------:|
| <img src="https://github.com/user-attachments/assets/051a662d-ec52-4cc2-b746-be0d36d8c041" width="90%">         |  <video src="https://github.com/user-attachments/assets/7dcf295d-4356-4490-b281-ec1d5d69866b" width="100%"> </video>        |
|  <img src="https://github.com/user-attachments/assets/c385a11f-60c7-4919-b0f1-bc5e715f673c" width="90%">         |       <video src="https://github.com/user-attachments/assets/73315093-5c58-4b2c-8231-154f36124f76" width="100%"> </video>        |

To generate the above videos, please refer to the [Image-to-Video Inference](#run-image-to-video-inference) section.

## 🚀 Quick Start

### Checkpoints

Please download all checkpoints and convert them into MindSpore checkpoints following this [instruction](./ckpts/README.md).

### Run VAE reconstruction

To run a video reconstruction task using the CausalVAE, please use the following command:
```bash
python scripts/run_vae.py \
    --video-path "path/to/input_video.mp4" \
    --rec-path "reconstructed_video.mp4" \
    --height 336 \
    --width 336 \
    --num-frames 65 \
```
The reconstructed video is saved under `./save_samples/`. To run reconstruction on an input image or a input folder of videos, please refer to `scripts/vae/recon_image.sh` or `scripts/vae/recon_video_folder.sh`.


<!-- ### Run Text-to-Video Inference

To run the text-to-video inference on a single prompt, please use the following command:
```bash
bash scripts/hyvideo/run_t2v_sample.sh
```
If you want change to another prompt, please set `--prompt` to the new prompt. -->

### Run Image-to-Video Inference

To run the image-to-video inference on a single prompt, please use the following command:
```bash
bash scripts/hyvideo-i2v/run_sample_image2video_stability.sh # or run_sample_image2video_dynamic.sh
```
If you want change to another prompt, please set `--prompt` to the new prompt.

To run image-to-video inference with LoRA weight, please refer to `scripts/hyvideo-i2v/run_sample_image2video_lora.sh`.


## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://arxiv.org/abs/2412.03603), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
