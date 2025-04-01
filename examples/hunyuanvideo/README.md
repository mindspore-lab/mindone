# Hunyuan Video: A Systematic Framework For Large Video Generation Model

This is a **MindSpore** implementation of [HunyuanVideo](https://arxiv.org/abs/2412.03603). It contains the code for **training** and **inference** of HunyuanVideo and 3D CausalVAE.


## 📑 Development Plan

Here is the development plan of the project:

- CausalVAE:
    - [x] Inference
    - [x] Evalution
    - [x] Training
- HunyuanVideo (13B):
    - [x] Inference
    - [x] Sequence Parallel (Ulysses SP)
    - [x] Training stage 1: T2I 256px
    - [ ] Training stage 2: T2I  256px 512px (buckts)
    - [ ] Training stage 3: T2I/V up to 720x1280x129 (buckts)
    - [ ] LoRA finetune



## 📦 Requirements


<div align="center">

| MindSpore | Ascend Driver |  Firmware   | CANN toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
|   2.4.1   |  24.1.RC2     | 7.5.0.2.220 |  8.0.RC3.beta1      |

</div>

1. Install
   [CANN 8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1)
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


## 🚀 Quick Start

### Checkpoints

Please download all checkpoints and convert them into MindSpore checkpoints following this [instruction](./docs/checkpoints_docs.md).

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


### Run Text-to-Video Inference

To run the text-to-video inference on a single prompt, please use the following command:
```bash
bash scripts/hyvideo/run_t2v_sample.sh
```
If you want change to another prompt, please set `--prompt` to the new prompt.

If you want to run T2V inference using sequence parallel (Ulysses SP), please use `scripts/hyvideo/run_t2v_sample_sp.sh`. You can revise the SP size using `--sp-size`, which should be aligned with `ASCEND_RT_VISIBLE_DEVICES`, `--worker_num` and `--local_worker_num`. See more usage information about `msrun` from this [website](https://www.mindspore.cn/docs/en/r2.4.10/model_train/parallel/msrun_launcher.html).

### Run Image-to-Video Inference

Please find more information about HunyuanVideo Image-to-Video Inference at this [url](https://github.com/mindspore-lab/mindone/tree/master/examples/hunyuanvideo-i2v).

## 🔑 Training

In this section, we provide instructions on how to train the HunyuanVideo model. For training instructions for 3D-VAE, please refer to [this document](docs/3d_vae_docs.md).

### Dataset Preparation

To prepare the dataset for training HunyuanVideo, please refer to the [dataset format](./docs/dataset_docs.md).

### Extract Text Embeddings

You need to extract the text embeddings for the train and validation dataset using the following command respectively:
```bash
python scripts/run_text_encoder.py \
  --data-file-path /path/to/caption.csv \
  --output-path /path/to/text_embed_folder \
```
Please refer to `scripts/text_encoder/run_text_encoder.sh` for more details.

Please also extract the text embedding for an empty string, because it will be used during training when the prompt is dropped randomly.
```bash
python scripts/run_text_encoder.py \
  --prompt "" \
  --output-path /path/to/text_embed_folder \
```

### Extract VAE Lantent Cache

To extract the latent cache for the train and validation dataset, please use the following command respectively:
```bash
WIDTH=256
HEIGHT=256
python scripts/run_vae_latent_cache.py \
  --input-video-dir /path/to/video_folder \
  --latent-cache-dir /path/to/latent_cache_folder \
  --height $HEIGHT \
  --width $WIDTH \
```
Please set the `--input-video-dir` and `--latent-cache-dir` to specify the input video folder and the output latent cache folder.

### Distributed Training

To run stage 1 (256px) trainig with HunyuanVideo (13B) on multiple NPUs, we use ZeRO3 and data parallelism with the following script:

```bash
bash scripts/hyvideo/train_t2v_stage1.sh
```

To run stage 2 (512px) trainig with HunyuanVideo (13B) on multiple NPUs using Ulysses SP, please run the following command:

```bash
bash scripts/hyvideo/train_t2v_stage2.sh
```

For the finetuning experiment with a small dataset, please refer to `scripts/hyvideo/train_t2v_256x256x29_finetune.sh`.

## 📈 Evaluation

### VAE Reconstruction Evaluation

To run video reconstruction on a folder of videos, please refer to `scripts/vae/recon_video_folder.sh`.

To evaluate the PSNR score between the real and the reconsturcted videos, you may use `scripts/eval/script/cal_psnr.sh`.

To specify the mindspore checkpoint path, please use `--ms-checkpoint`. See more usage information using `python scripts/run_vae.py --help`.

### Text-to-Video Evalution

After training, the checkpoint shards will be saved under `output/experiment_dir/`. The folder structure is as follows:
```bash
output/experiment_dir/
├───rank_0
│   └───ckpt/
│       └───HYVideo-T-2-cfgdistill-s10000.ckpt
├───rank_1
│   └───ckpt/
│       └───HYVideo-T-2-cfgdistill-s10000.ckpt
...
└───rank_7
    └───ckpt/
        └───HYVideo-T-2-cfgdistill-s10000.ckpt
```


To run Text-to-Video evaluation with the saved checkpoint shards, please refer to `scripts/hyvideo/run_t2v_sample_multi.sh`. You need to change the `--dit-weight` to the saved checkpoint path, for example:
```bash
--dit-weight "output/experiment_dir/rank_*/ckpt/HYVideo-T-2-cfgdistill-s10000.ckpt"
```

A wildcard pattern `*` is needed to match any rank ids of the checkpoint shards.

### 3D-VAE Training

See the training tutorial of Causal 3D-VAE from [here](docs/3d_vae_docs.md)


## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://arxiv.org/abs/2412.03603), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
