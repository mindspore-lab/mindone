# Getting Started with SDXL

This document provides a brief introduction to the usage of built-in command-line tools in SDXL.

## Dependency

- mindspore 2.1.0
- openmpi 4.0.3 (for distributed mode)

To install the dependency, please run

```shell
pip install -r requirements.txt
```

## Preparation

### Convert Pretrained Checkpoint

We provide a script for converting pre-trained weight from `.safetensors` to `.ckpt` in `tools/model_conversion/convert_weight.py`.

step1. Download the [Official](https://github.com/Stability-AI/generative-models) pre-train weights [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [SDXL-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) from huggingface.

step2. Convert weight to MindSpore `.ckpt` format and put it to `./checkpoints/`.

```shell
cd tools/model_conversion

# convert sdxl-base-1.0 model
python convert_weight.py \
  --weight_safetensors /PATH TO/sd_xl_base_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_base_1.0_ms.ckpt \
  --key_torch torch_key_base.yaml \
  --key_ms mindspore_key_base.yaml

# convert sdxl-refiner-1.0 model
python convert_weight.py \
  --weight_safetensors /PATH TO/sd_xl_refiner_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_refiner_1.0_ms.ckpt \
  --key_torch torch_key_refiner.yaml \
  --key_ms mindspore_key_refiner.yaml
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

1. (Recommend) Run with interactive visualization:

```shell
# (recommend) run with streamlit
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
streamlit run demo/sampling.py --server.port <your_port>
```

2. Run with other methods:

<details close>

```shell
# run sdxl-base txt2img without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --device_target Ascend

# run sdxl-refiner img2img without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task img2img \
  --config configs/inference/sd_xl_refiner.yaml \
  --weight checkpoints/sd_xl_refiner_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --img /PATH TO/img.jpg \
  --device_target Ascend

# run pipeline without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --add_pipeline True \
  --pipeline_config configs/inference/sd_xl_refiner.yaml \
  --pipeline_weight checkpoints/sd_xl_refiner_1.0_ms.ckpt \
  --sd_xl_base_ratios "1.0_768" \
  --device_target Ascend

# run lora(unmerge weight) without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_lora.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt,SDXL-base-1.0_2000_lora.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --device_target Ascend
```

</details>

### Invisible Watermark Detection

To be supplemented


## Training and Fine-Tuning

⚠️ This function is experimental. The script fine-tunes the whole model and often the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyper-parameters to get the best result on your dataset.

We are providing example training configs in `configs/training`. To launch a training, run

```shell
# sdxl-base lora fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_base_finetune_lora.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \
  --device_target Ascend

# sdxl-refiner lora fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_refiner_finetune_lora.yaml \
  --weight checkpoints/sd_xl_refiner_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \
  --device_target Ascend

# run with multiple NPU/GPUs
mpirun --allow-run-as-root -n 8 python train.py \
  --config /PATH TO/config.yaml \
  --weight /PATH TO/weight.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \
  --is_parallel True \
  --device_target <YOUR DEVCIE>
```
