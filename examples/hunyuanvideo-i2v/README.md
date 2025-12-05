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
|   2.6.0    |    24.1.RC3    | 7.5.T11.0.B088 |   8.1.RC1     |
|   2.7.0    |    24.1.RC3    | 7.5.T11.0.B088 |   8.2.RC1     |

</div>

1. Install
   [CANN 8.1.RC1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.1.RC1) or [CANN 8.2.RC1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)
   and MindSpore according to the [official instructions](https://www.mindspore.cn/install).
2. Install requirements
    ```shell
    pip install -r requirements.txt
    ```
    In case `decord` package is not available, try `pip install eva-decord`.
    For EulerOS, instructions on ffmpeg and decord installation are as follows.

    <details onclose>
    <summary>How to install ffmpeg and decord</summary>

    ```
    1. install ffmpeg 4, referring to https://ffmpeg.org/releases
        wget https://ffmpeg.org/releases/ffmpeg-4.0.1.tar.bz2 --no-check-certificate
        tar -xvf ffmpeg-4.0.1.tar.bz2
        mv ffmpeg-4.0.1 ffmpeg
        cd ffmpeg
        ./configure --enable-shared         # --enable-shared is needed for sharing libavcodec with decord
        make -j 64
        make install

    2. install decord, referring to https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source
        git clone --recursive https://github.com/dmlc/decord
        cd decord
        rm build && mkdir build && cd build
        cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
        make -j 64
        make install
        cd ../python
        python3 setup.py install --user
    ```

    </details>

## 🎥 Demo

The following videos are generated based on MindSpore and Ascend Atlas 800T A2 machines.

|  Reference Image | Generated Video |
|:----------------:|:----------------:|
| <img src="https://github.com/user-attachments/assets/051a662d-ec52-4cc2-b746-be0d36d8c041" width="90%">         |  <video src="https://github.com/user-attachments/assets/7dcf295d-4356-4490-b281-ec1d5d69866b" width="100%"> </video>        |
|  <img src="https://github.com/user-attachments/assets/c385a11f-60c7-4919-b0f1-bc5e715f673c" width="90%">         |       <video src="https://github.com/user-attachments/assets/73315093-5c58-4b2c-8231-154f36124f76" width="100%"> </video>        |

|  Reference Image | Generated Video (w. LoRA) |
|:----------------:|:----------------:|
| <img src="https://github.com/user-attachments/assets/e28140e7-2038-43a6-8ce8-9d738649c356" width="90%">          |  <video src="https://github.com/user-attachments/assets/fc0638f8-e342-4994-96d6-4234d692cb5e" width="100%"> </video>        |
|  <img src="https://github.com/user-attachments/assets/058ce139-1f2c-4aac-9700-dbcd8c847117" width="90%">         |       <video src="https://github.com/user-attachments/assets/e6bba358-2385-499e-ae81-67176cc8a806" width="100%"> </video>        |

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

## Performance

### Inference

The following experiments are tested on Ascend Atlas 800T A2 machines with **mindspore 2.7.0 pynative mode**.

| model | cards | batch size | resolution | num of frames | num of steps | step time (sec) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|  HYVideo-T/2 | 1 | 1  | 720p |  129 | 50 | 86.02 |

The following experiments are tested on Ascend Atlas 800T A2 machines with **mindspore 2.6.0 pynative mode**.

| model | cards | batch size | resolution | num of frames | num of steps | step time (sec) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|  HYVideo-T/2 | 1 | 1  | 720p |  129 | 50 | 85.94 |



## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://arxiv.org/abs/2412.03603), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
