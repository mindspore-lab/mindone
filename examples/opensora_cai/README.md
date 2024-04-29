A mindspore implementation of [OpenSora](https://github.com/hpcaitech/Open-Sora) from hpcaitech.

## TODOs
- [x] STDiT implementation and inference
    - [x] refactor Masked MultiHeadCrossAttention without xformers
    - [ ] more efficient masking and attention computation for text tokens with dynamic length.
- [ ] Text-to-video generation pipeline (to be refactored)
    - [x] video generation in FP32/FP16 precision on GPUs: 256x256x16, 512x512x16
    - [x] video generation in FP32/FP16 precision on Ascends: 256x256x16, 512x512x16
    - [ ] Mixed precision optimization (BF16)  on Ascend
    - [x] Flash attention optimization on Ascend
- [ ] Training
    - [x] Text embedding-cached STDiT training on GPUs and Ascends
        - [x] small dataset
        - [x] train with long frames, up to **512x512x300**
    - [ ] Training with online T5-embedding
    - [ ] Train in BF16 precision
    - [ ] Zero2 and sequence-parallel training


## Requirements

```
pip install -r requirements.txt
```

MindSpore version: >= 2.2.12

To enable flash attention, please use mindspore>=2.3-20240422.

## Preparation

Prepare the model checkpoints of T5, VAE, and STDiT and put them under `models/` folder as follows

- T5: [ms checkpoints download link](https://download-mindspore.osinfra.cn/toolkits/mindone/text_encoders/deepfloyd_t5_v1_1_xxl/)

    Put them under `models/t5-v1_1-xxl` folder. Rename `t5_v1_1_xxl-d35f27a3.ckpt` to `model.ckpt` if error raised.

- VAE: [safetensor download link](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/tree/main)

    Convert to ms checkpoint: `python tools/convert_pt2ms.py --src /path/to/vae-ft-mse-840000-ema-pruned.safetensors --target models/sd-vae-ft-mse.ckpt`

    For `sd-vae-ft-ema`, run:
    ```
    python tools/vae_converter.py --source /path/to/sd-vae-ft-ema/diffusion_pytorch_model.safetensors --target models/sd-vae-ft-ema.ckpt
    ```

- STDiT: [pth download link](https://huggingface.co/hpcai-tech/Open-Sora/tree/main)

    Convert to ms checkpoint: `python tools/convert_pt2ms.py --src /path/to/OpenSora-v1-16x256x256.pth --target models/OpenSora-v1-16x256x256.ckpt`

- PixArt-α: [pth download link](https://download.openxlab.org.cn/models/PixArt-alpha/PixArt-alpha/weight/PixArt-XL-2-512x512.pth)  (for training only)

    Convert to ms checkpoint: `python tools/convert_pt2ms.py --src /path/to/PixArt-XL-2-512x512.pth --target models/PixArt-XL-2-512x512.ckpt`
    It will be used for better model initialziation.

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

<p float="left">
<img src=https://github.com/SamitHuang/mindone/assets/8156835/b3780fdc-6b14-425f-a26b-9564c0f492a2 width="24%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/16fe702e-31ac-4651-bce8-876aec7212f9 width="24%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/864a8a41-fd68-4343-ae71-6c25b28f1e6b width="24%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/fde06430-9fd8-48b2-b8ba-890f60d47534 width="24%" />
</p>

<p float="left">
<img src=https://github.com/SamitHuang/mindone/assets/8156835/bd63e6fb-c498-4673-b260-505a168b0efa width="24%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/89136878-ae3b-4d16-a8f9-71ee9ecc420e width="24%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/ac42f81c-519b-4717-9f04-5585e0509323 width="24%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/66a5c543-9693-4129-b2af-edef67abff79 width="24%" />
</p>

(source prompts from [here](https://github.com/hpcaitech/Open-Sora/blob/main/assets/texts/t2v_samples.txt))


## Training

### 1. Generate T5 embeddings
```
python infer_t5.py \
    --csv_path ../videocomposer/datasets/webvid5/video_caption.csv \
    --output_dir ../videocomposer/datasets/webvid5 \
```

After running, the text embeddings saved as npz file for each caption will be in `output_dir`

Please change `csv_path` to your video-caption annotation file accordingly.

### 2. Generate VAE embeddings (Optional)
```
python infer_vae.py \
    --csv_path ../videocomposer/datasets/webvid5/video_caption.csv \
    --output_dir ../videocomposer/datasets/webvid5_vae_256x256 \
    --vae_checkpoint models/sd-vae-ft-ema.ckpt \    # or sd-vae-ft-mse.ckpt
    --video_folder ../videocomposer/datasets/webvid5  \
    --image_size 256 \
```

After running, the vae latents saved as npz file for each video will be in `output_dir`.


### 3. Train STDiT

```
python train_t2v.py --config configs/train/stdit_256x256x16.yaml \
    --csv_path "../videocomposer/datasets/webvid5/video_caption.csv" \
    --video_folder "../videocomposer/datasets/webvid5" \
    --text_embed_folder "../videocomposer/datasets/webvid5" \
```

To to enable training with the cached vae latents, please append `--vae_latent_folder "../videocomposer/datasets/webvid5_vae_256x256"`.

Please change `csv_path`,`video_folder`, `embed_folder` according to your data location.

For detailed usage, please check `python train_t2v.py -h`

Note that the training precision is under continuous optimization.


#### Notes about MindSpore 2.3

Training on MS2.3 allows much better performance with its new feautres (such as kbk and dvm)

To enable kbk mode on ms2.3, please set
```
export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1

```

To improve training perforamnce, you may append `--enable_dvm=True` to the training command.

Here is an example for training on MS2.3:
```
export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1

python train_t2v.py --config configs/train/stdit_256x256x16.yaml \
    --csv_path "../videocomposer/datasets/webvid5/video_caption.csv" \
    --video_folder "../videocomposer/datasets/webvid5" \
    --text_embed_folder "../videocomposer/datasets/webvid5" \
    --enable_dvm=True \
```




### Evaluate
To evaluate the training result:

```
python sample_t2v.py \
    --config configs/inference/stdit_256x256x16_webvid.yaml \
    --checkpoint /path/to/your_trained_model.ckpt \
```

You may change the source captions in the config yaml file (key `captions:`)
