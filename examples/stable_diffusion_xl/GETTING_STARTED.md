# Getting Started with SDXL

This document provides a brief introduction to the usage of built-in command-line tools in SDXL.

## Preparation

### Convert Pretrained Checkpoint

We provide a script for converting pre-trained weight from `.safetensors` to `.ckpt` in `scripts/convert/convert_weight.py`.

step1. Download [Official pre-train weights](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors) from huggingface.

step2. Convert Weight to MindSpore `.ckpt` format.

```shell
python scripts/convert/convert_weight.py \
  --weight_safetensors "./checkpoints/sd_xl_base_1.0.safetensors" \
  --weight_ms "./checkpoints/sd_xl_base_1.0_ms.ckpt"
```

### Dataset Preparation for Fine-Tuning (Optional)

The text-image dataset for fine-tuning should follow the file structure below

<details onclose>

```text
dir
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── img_txt.csv
```

img_txt.csv is the annotation file in the following format
```text
dir,text
img1.jpg,a cartoon character with a potted plant on his head
img2.jpg,a drawing of a green pokemon with red eyes
img3.jpg,a red and white ball with an angry look on its face
```

For convenience, we have prepared two public text-image datasets obeying the above format.

- [pokemon-blip-caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 833 pokemon-style images with BLIP-generated captions.
- [Chinese-art blip caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 100 Chinese art-style images with BLIP-generated captions.

To use them, please download `pokemon_blip.zip` or `chinese_art_blip.zip` from the [openi dataset website](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets). Then unzip.

</details>


## Inference

We provide a demo for text-to-image sampling in `demo/sampling_without_streamlit.py` and `demo/sampling.py` with [streamlit](https://streamlit.io/).

After obtaining the weights, place them into checkpoints/. Next, start the demo using

```shell
# run with streamlit
streamlit run demo/sampling.py --server.port <your_port>

# run without streamlit
python demo/sampling_without_streamlit.py \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# run without streamlit (LoRA unmerge weight)
python demo/sampling_without_streamlit.py \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --weight "checkpoints/sd_xl_base_1.0_ms.ckpt,SDXL-base-1.0_2000_lora.ckpt" \
  --config config/training/sd_xl_base_finetune_lora.yaml
```

<br>

Examples (sample 40 steps by EulerEDMSampler):

<div align="center">
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/68d132e1-a954-418d-8cb8-5be4d8162342" width="320" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/9f0d0d2a-2ff5-4c9b-a0d0-1c744762ee92" width="320" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/dbaf0c77-d8d3-4457-b03c-82c3e4c1ba1d" width="320" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/f52168ef-53aa-4ee9-9f17-6889f10e0afb" width="320" />
</div>
<p align="center">
<font size=3>
<em> Fig1: "Vibrant portrait painting of Salvador Dalí with a robotic half face." </em> <br>
<em> Fig2: "A capybara made of voxels sitting in a field." </em> <br>
<em> Fig3: "Cute adorable little goat, unreal engine, cozy interior lighting, art station, detailed’ digital painting, cinematic, octane rendering." </em> <br>
<em> Fig4: "A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grass in front of the Sydney Opera House holding a sign on the chest that says "SDXL"!." </em>
</font>
</p>

<br>

### Invisible Watermark Detection

To be supplemented


## Training and Fine-Tuning

We are providing example training configs in `configs/training`. To launch a training, run

```shell
# vanilla fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_base_finetune.yaml \
  --data_path /path_to/dataset/ \

# lora fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_base_finetune_lora.yaml \
  --data_path /path_to/dataset/ \

# run with multiple NPU/GPUs
mpirun --allow-run-as-root -n 8 python train.py \
  --config /path_to/config.yaml \
  --data_path /path_to/dataset/ \
  --is_parallel True \
  --device_target <your_device>
```
