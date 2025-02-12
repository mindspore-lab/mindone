# Hunyuan Video


## Quick Start

### Installation

### Checkpoints

Please download all checkpoints and convert them into MindSpore checkpoints following this [instruction](./ckpts/README.md).

### Run VAE reconstruction

To run a video reconstruction using the CausalVAE, please use the following command:
```bash
python scripts/run_vae.py \
    --video-path "path/to/input_video.mp4" \
    --output-path "path/to/output_directory" \
    --rec-path "reconstructed_video.mp4" \
    --height 336 \
    --width 336 \
    --num-frames 65 \
```
The reconstructed video is saved under `./save_samples/`. To run reconstruction on an input image or a input folder of videos, please refer to `scripts/vae/recon_image.sh` or `scripts/vae/recon_video_folder.sh`.


## 🔑 Training

### Dataset Preparation

To prepare the dataset for training HuyuanVideo, please refer to the [dataset format](./hyvideo/dataset/README.md).

### Extract Text Embeddings

```bash
python scripts/run_text_encoder.py \
  --data-file-path /path/to/caption.csv \
  --output-path /path/to/text_embed_folder \
```
Please refer to `scripts/text_encoder/run_text_encoder.sh`. More details can be found by `python scripts/run_text_encoder.py --help`.

### Distributed Training

To train HunyuanVideo (13B) on multiple NPUs, we use ZeRO3 and data parallelism with the following script:

```bash
bash scripts/train_t2v_zero3.sh
```


### Run Text-to-Video Inference




### Run Image-to-Video Inference


## Train


## Evaluation


### VAE Evaluation

### Video Reconstruction and Evalution

To run video reconstruction using 3D-VAE, please use the following command:

```bash
python hyvideo/rec_video.py \
  --video_path input_video.mp4 \
  --rec_path rec.mp4 \
  --height 360 \
  --width 640 \
  --num_frames 33 \
```

The reconstructed video will be saved under `./samples/`.

To run video reconstruction on a folder of videos, please replace the script with `hyvideo/rec_video_folder.py` and use `--real_video_dir` to parse the video folder path.

To evaluate the reconsturcted videos, you may use the  `hyvideo/eval/scripts/cal_psnr.sh` script.

TODOs:
- [ ] For simplicity, remove `rec_video_folder.py` and allow evaluate a video folder in `rec_video.py` (e.g. `--real_video_dir`), and evaluate PSNR when video reconstruction finished.

### 3D-VAE Training

coming soon...


## Embedding Cache

### Text embedding cache

To generate text embeddings given a dataset annotation file in JSON format, please use the following command:

```bash
python scripts/run_text_encoder.py \
    --data-file-path /path/to/caption.json \
    --output-path /path/to/text_embed_folder \
```

Please refer to [dataset format](hyvideo/dataset/README.md) to setup the json file.  A shell script `scripts/text_encoder/run_text_encoder.sh` is provided as well.


If you just want to generate text embedding for a single prompt, you can run like:
```bash
python scripts/run_text_encoder.py \
    --prompt "A cat walks on the grass, realistic style." \
    --output-path /path/to/text_embed_folder \
```

The generated npz file, which contains the prompt embedding using clip and llm and the prompt mask, will be saved in the current folder.



### Video embedding cache


## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://arxiv.org/abs/2412.03603), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
