A mindspore implementation of [OpenSora](https://github.com/hpcaitech/Open-Sora) from hpcaitech.

## TODOs
- [x] STDiT implementation and inference
    - [x] refactor Masked MultiHeadCrossAttention without xformers
    - [ ] more efficient masking and attention computation for text tokens with dynamic length.
- [ ] Text-to-video generation pipeline (to be refactored)
    - [x] video generation in FP32 precision on GPUs: 256x256x16, 512x512x16
    - [ ] video generation in FP32 precision on Ascends
    - [ ] Mixed precision optimization (BF16)  on Ascend
    - [ ] Flash attention optimization on Ascend
- [ ] Training  (**Experimental**)
    - [x] Text embedding-cached STDiT training on GPUs and Ascends
        - [x] small dataset
        - [ ] train with long frames
    - [ ] Training with online T5-embedding
    - [ ] Train in BF16 precision
    - [ ] Zero2 and sequence-parallel training


## Requirements

```
pip install -r requirements.txt
```

MindSpore version: >= 2.2.12 


## Preparation

Prepare the model checkpoints of T5, VAE, and STDiT and put them under `models/` folder as follows

- T5: [ms checkpoints download link](https://download-mindspore.osinfra.cn/toolkits/mindone/text_encoders/deepfloyd_t5_v1_1_xxl/)

    Put them under `models/t5-v1_1-xxl` folder. Rename `t5_v1_1_xxl-d35f27a3.ckpt` to `model.ckpt` if error raised.

- VAE: [safetensor download link](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/tree/main)

    Convert to ms checkpoint: `python tools/convert_pt2ms.py --src /path/to/vae-ft-mse-840000-ema-pruned.safetensors --target models/sd-vae-ft-mse.ckpt`

- STDiT: [pth download link](https://huggingface.co/hpcai-tech/Open-Sora/tree/main)

    Convert to ms checkpoint: `python tools/convert_pt2ms.py --src /path/to/OpenSora-v1-16x256x256.pth --target models/OpenSora-v1-16x256x256.ckpt`


## Inference

To generate video conditioning on captions:
```
python sample_t2v.py --config configs/inference/stdit_256x256x16.yaml
```
> By default, FP32 is used to ensure the best precision. Nan values may incur in stdit forward pass using fp16, resulting in dark videos.

- To run on GPU, append
`--device_target GPU`

- To run with cached t5 embedding, append
`--embed_path outputs/t5_embed.npz`

For more usage, please run `python sample_t2v.py -h`

To get t5 embedding for a few captions:
```
python infer_t5.py --config configs/inference/stdit_256x256x16.yaml --output_path=outputs/t5_embed.npz
```

Here are some generation results in 256x256 resolution.

![0](https://github.com/SamitHuang/mindone/assets/8156835/f24f6ecb-84e3-4651-870a-ad562b30fc6e)
![1](https://github.com/SamitHuang/mindone/assets/8156835/65268e97-6bcb-4b23-a3c9-3f0c74b0b624)

!!NOTE: the inference precision is still under optimization on Ascend.

## Training

### 1. Generate T5 embeddings
```
python infer_t5.py \
    --csv_path ../videocomposer/datasets/webvid5/video_caption.csv \
    --output_dir ../videocomposer/datasets/webvid5 \
```

After running, the text embeddings saved as npz file for each caption will be in `output_dir`

Please change `csv_path` to your video-caption annotation file accordingly.

### 2. Train STDiT

```
python train_t2v.py --config configs/train/stdit_256x256x16.yaml \
    --csv_path "../videocomposer/datasets/webvid5/video_caption.csv" \
    --video_folder "../videocomposer/datasets/webvid5" \
    --embed_folder "../videocomposer/datasets/webvid5" \
```

Please change `csv_path`,`video_folder`, `embed_folder` according to your data location.

For detailed usage, please check `python train_t2v.py -h`

!!NOTE: currently, the training script is **experimental** and is mainly used to evaluate the training performance (i.e. max frames and step training time).

It's under continuous optimization.
