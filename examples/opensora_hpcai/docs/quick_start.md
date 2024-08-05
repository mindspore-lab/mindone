A mindspore implementation of [OpenSora](https://github.com/hpcaitech/Open-Sora) from HPC-AI Tech.

## TODOs
- [x] STDiT implementation and inference
    - [x] refactor Masked MultiHeadCrossAttention without xformers
    - [ ] more efficient masking and attention computation for text tokens with dynamic length.
- [x] Text-to-video generation pipeline (to be refactored)
    - [x] video generation in FP32/FP16 precision on GPUs: 256x256x16, 512x512x16
    - [x] video generation in FP32/FP16 precision on Ascends: 256x256x16, 512x512x16
    - [x] Mixed precision optimization (BF16) on Ascend
    - [x] Flash attention optimization on Ascend
- [x] Image/Video-to-Video generation pipeline (to be refactored)
    - [x] Video generation in FP32 precision on Ascend 910*.
    - [x] Mixed precision optimization (FP16 and BF16) on Ascend.
- [ ] Training
    - [x] Text embedding-cached STDiT training on GPUs and Ascends
        - [x] small dataset
        - [x] train with long frames, up to **512x512x300**
    - [x] Masked frames training pipeline
    - [ ] Training with online T5-embedding
    - [x] Train in BF16 precision
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

- VAE:

  **OpenSora v1.2**:
  Download from [Hugging Face](https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2) and convert to MS checkpoint:
  ```shell
  python tools/convert_vae_3d.py --src path/to/OpenSora-VAE-v1.2/model.safetensors --target models/OpenSora-VAE-v1.2/model.ckpt
  ```

  **OpenSora v1.1 and below**:
  Download from [Hugging Face](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/tree/main) and convert to MS checkpoint:
  ```shell
  python tools/convert_pt2ms.py --src /path/to/vae-ft-mse-840000-ema-pruned.safetensors --target models/sd-vae-ft-mse.ckpt
  ```
  For `sd-vae-ft-ema`, run:
  ```shell
  python tools/convert_vae.py --src /path/to/sd-vae-ft-ema/diffusion_pytorch_model.safetensors --target models/sd-vae-ft-ema.ckpt
  ```

- STDiT:
    - OpenSora v1.2: [download](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3)
    - OpenSora v1.1: [stage2](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage2) or [stage3](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3)
    - OpenSora v1: [pth download link](https://huggingface.co/hpcai-tech/Open-Sora/tree/main)

    Convert to ms checkpoint: `python tools/convert_pt2ms.py --src /path/to/checkpoint --target models/checkpoint_name.ckpt`

- PixArt-α: [pth download link](https://download.openxlab.org.cn/models/PixArt-alpha/PixArt-alpha/weight/PixArt-XL-2-512x512.pth) (for training only)

    Convert to ms checkpoint: `python tools/convert_pt2ms.py --src /path/to/PixArt-XL-2-512x512.pth --target models/PixArt-XL-2-512x512.ckpt`
    It will be used for better model initialization.


## Inference

### Text-to-Video
Configuration files can be found in the desired OpenSora version folder:

**OpenSora v1.2**
```shell
python scripts/inference.py --config configs/opensora-v1-2/inference/sample_t2v.yaml
```

**OpenSora v1.1**
```shell
python scripts/inference.py --config configs/opensora-v1-1/inference/sample_t2v.yaml
```

**OpenSora v1**
```shell
python scripts/inference.py --config configs/opensora/inference/stdit_256x256x16.yaml
```

- To run on GPU, append
`--device_target GPU`

- To run with cached t5 embedding, append
`--embed_path outputs/t5_embed.npz`

For more usage, please run `python scripts/inference.py -h`

To get t5 embedding for a few captions:
```
python scripts/infer_t5.py--config configs/opensora/inference/stdit_256x256x16.yaml --output_path=outputs/t5_embed.npz
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


### Image/Video-to-Video (OpenSora v1.1 and above)

Conditioning on images and videos in OpenSora is based on a frame masking strategy.
Specifically, conditioning frames are unmasked and assigned a timestep of 0,
while other frames are assigned a timestep _t_. An example is shown below:

<p align="center"><img alt="mask strategy" src="https://github.com/mindspore-lab/mindone/assets/16683750/0cf5b478-288f-4f53-906d-26fb7b93182c" width="750"/></p>

To generate videos conditioned on images and videos, you will need to specify the following parameters in the
[config file](../configs/opensora-v1-1/inference/sample_iv2v.yaml):

```yaml
loop: 2
condition_frame_length: 4
captions:
  - "In an ornate, historical hall, a massive tidal wave peaks and begins to crash. Two surfers, seizing the moment, skillfully navigate the face of the wave."
mask_strategy:
  - "0"
reference_path:
  - "assets/wave.png"
```

<p align="center"><img alt="mask strategy config" src="https://github.com/mindspore-lab/mindone/assets/16683750/734c5dc6-13ba-45e4-b3f2-6da9f523296c" width="750"/></p>

Where:  
`loop` (integer): Specifies the number of iterations for the video generation process, conditioning on references or videos generated in previous loops (if no reference is provided).  
`condition_frame_length` (integer): The number of frames from the previous loop to use for conditioning.  
`captions` (list of string): A list of captions for conditioning. Different captions can be assigned to each loop by separating them with `|X|`, where `X` represents the loop number.
`mask_strategy` (list of string): A list of mask strategies in the format of six numbers separated by comma:
- First number: The loop index to which the current mask strategy applies.
- Second number: The index of the condition image or video in the `reference_path`.
- Third number: The reference starting frame number to use for conditioning.
- Fourth number: The position at which to insert reference frames into the generated video.
- Fifth number: The number of frames used for conditioning.
- Sixth number: The intensity of editing conditioning frames, where 0 means no edit and 1 means a complete edit.

`reference_path` (list of string): A list of reference paths corresponding to each caption.
Each loop can have a different reference path. In this case, the reference path must be separated by semicolon.

The output video's length will be `loop * (num_frames - condition_frame_length) + condition_frame_length`.

To generate a video with conditioning on images and videos, execute the following command:
```shell
python scripts/inference.py --config configs/opensora-v1-1/inference/sample_iv2v.yaml --ckpt_path /path/to/your/opensora-v1-1.ckpt
```

## Training

### 1. Generate T5 embeddings (Required)
```shell
python scripts/infer_t5.py\
    --csv_path ../videocomposer/datasets/webvid5/video_caption.csv \
    --output_dir ../videocomposer/datasets/webvid5 \
    --model_max_length 200 # For OpenSora v1.1
```

> [!WARNING]
> OpenSora v1 requires text embedding sequence length of 120.  
> OpenSora v1.1 requires text embedding sequence length of 200.  
> OpenSora v1.2 requires text embedding sequence length of 300.

After running, the text embeddings saved as npz file for each caption will be in `output_dir`

Please change `csv_path` to your video-caption annotation file accordingly.

### 2. Generate VAE embeddings (Optional)
```
python scripts/infer_vae.py\
    --csv_path ../videocomposer/datasets/webvid5/video_caption.csv \
    --output_path ../videocomposer/datasets/webvid5_vae_256x256 \
    --vae_checkpoint models/sd-vae-ft-ema.ckpt \    # or sd-vae-ft-mse.ckpt
    --video_folder ../videocomposer/datasets/webvid5  \
    --image_size 256 \
```

After running, the vae latents saved as npz file for each video will be in `output_dir`.

For parallel inference, please refer to `scripts/run/run_infer_vae_parallel.sh`


### 3. Train STDiT

```
python scripts/train.py --config configs/opensora/train/stdit_256x256x16.yaml \
    --csv_path YOUR_CSV_PATH \
    --video_folder YOUR_VIDEO_FOLDER \
    --text_embed_folder YOUR_TEXT_EMBED_FOLDER \
```

To enable training with the cached vae latents, please append `--vae_latent_folder YOUR_VAE_LATENT_FOLDER`.

Please change `csv_path`,`video_folder`, `text_embed_folder` according to your data location.

For detailed usage, please check `python scripts/train.py -h`

> [!NOTE]
> Training precision is under continuous optimization.


### 4. Multi-resolution Training with Buckets (OpenSora v1.1 and above)

OpenSora v1.1 and above support training with multiple resolutions, aspect ratios, and a variable number of frames.
To enable this feature, add the desired bucket configuration to the `yaml` config file
(see [train_stage1.yaml](../configs/opensora-v1-1/train/train_stage1.yaml) for an example).

The bucket configuration is a two-level dictionary formatted as `resolution: { num_frames: [ keep_prob, batch_size ] }`,
where:

- `resolution` specifies the resolution of a particular bucket.
- `num_frames` is the number of frames in the bucket.
- `keep_prob` is the probability of a video being placed into the bucket.
- `batch_size` refers to the batch size for the bucket.

The available resolutions and aspect ratios are predefined and can be found
in [aspect.py](../opensora/datasets/aspect.py).
The `keep_prob` parameter determines the likelihood of a video being placed into a particular bucket.
A bucket is selected based on the video resolution, beginning with the highest resolution that does not exceed the
video's own resolution.

The selection process considers only the maximum possible number of frames for each bucket,
meaning that the buckets are selected based on resolution alone.

> [!TIP]
> If you want longer videos to go into smaller resolution buckets, you can set the `keep_prob` to `0.0`,
> as shown in the example below:
> ```yaml
> bucket_config:
>   # Structure: resolution: { num_frames: [ keep_prob, batch_size ] }
>   # Setting [ keep_prob, batch_size ] to [ 0.0, 0 ] forces longer videos into smaller resolution buckets
>   "240p": {16: [1.0, 16], 32: [1.0, 8], 64: [1.0, 4], 128: [1.0, 2]}
>   "480p": {16: [1.0, 4], 32: [0.0, 0]}
>   "720p": {16: [0.5, 2]}
> ```
> With this configuration, videos with a length of 32 or more frames will be assigned to the `240p` bucket instead
> of `480p`.

#### Notes about MindSpore 2.3

Training on MS2.3 allows much better performance with `jit_level` (`"O0"`, `"O1"`, `"O2"`)

Here is an example for training on MS2.3:
```

python scripts/train.py --config configs/opensora/train/stdit_256x256x16.yaml \
    --csv_path "../videocomposer/datasets/webvid5/video_caption.csv" \
    --video_folder "../videocomposer/datasets/webvid5" \
    --text_embed_folder "../videocomposer/datasets/webvid5" \
    --jit_level="O1" \
```




### Evaluate
To evaluate the training result:

```
python scripts/inference.py \
    --config configs/opensora/inference/stdit_256x256x16_webvid.yaml \
    --ckpt_path /path/to/your_trained_model.ckpt \
```

You may change the source captions in the config yaml file (key `captions:`)
