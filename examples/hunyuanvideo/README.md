# Hunyuan Video


## Quick Start

### Installation

### Checkpoints

Please download all checkpoints and convert them into MindSpore checkpoints following this [instruction](./ckpts/README.md).

### Run text encoder

```bash
cd hyvideo
python run_text_encoder.py
```


### Run VAE reconstruction

To run a video reconstruction using the CausalVAE, please use the following command:
```bash
python hyvideo/rec_video.py \
  --video_path input_video.mp4 \
  --rec_path rec.mp4 \
  --height 360 \
  --width 640 \
  --num_frames 33 \
```
The reconstructed video is saved under `./samples/`. To run video reconstruction on a given folder of input videos, please see `hyvideo/rec_video_folder.py` for more information.


## 🔑 Training

### Dataset Preparation

To prepare the dataset for training HuyuanVideo, please refer to the [dataset format](./hyvideo/dataset/README.md).


### Training

To train HunyuanVideo, we use ZeRO3 and data parallelism to enable parallel training on 8 devices. Please run the following command:

```bash
bash scripts/train_t2v_zero3.sh
```


### Run Text-to-Video Inference




### Run Image-to-Video Inference


## Train


## Evaluation


### VAE Evaluation

To evaluate VAE's PNSR, please download MCL_JCV dataset from this [URL](https://mcl.usc.edu/mcl-jcv-dataset/), and place the videos under `datasets/MCL_JCV`.

Now, to run video reconstruction on a video folder, please run:

```bash
python hyvideo/rec_video_folder.py \
  --real_video_dir datasets/MCL_JCV \
  --generated_video_dir datasets/MCL_JCV_generated \
  --height 360 \
  --width 640 \
  --num_frames 33 \
```

Afterwards, you can evaluate the PSNR via:
```bash
bash hyvideo/eval/scripts/cal_psnr.sh
```

## Embedding Cache

### Text embedding cache

```bash
python hyvideo/run_text_encoder.py \
  --text_encoder_path /path/to/ckpt \
  --text_encoder_path_2 /path/to/ckpt \
  --data_file_path /path/to/caption.json \
  --output_path /path/to/text_embed_folder \
```

A shell script `scripts/run_text_encoder.sh` is provided as well.

### Video embedding cache


## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://arxiv.org/abs/2412.03603), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
