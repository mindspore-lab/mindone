# Lora Finetune

### Introduction

We provide the script `train.py` for lora finetuning of sdxl.

> Note: If you have network issues on downloading clip tokenizer, please manually download [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) from huggingface and change `version: openai/clip-vit-large-patch14` in `configs/training/sd_xl_base_finetune_lora_910b.yaml` to `version: your_path/to/clip-vit-large-patch14`

### Pretrained models

Download the official pre-train weights from huggingface, convert the weights from `.safetensors` format to Mindspore `.ckpt` format, and put them to `./checkpoints/` folder. Please refer to SDXL [weight_convertion.md](./weight_convertion.md) for detailed steps.


### lora finetune

```shell
# sdxl-base lora fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_base_finetune_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \
  --gradient_accumulation_steps 4 \
```

### inference, run lora(unmerge weight) without streamlit on Ascend

```shell
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_lora.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt,SDXL-base-1.0_2000_lora.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --device_target Ascend
```


### Benchmark

#### Training

| Model Name      |   Context      |  Method      | Global Batch Size x Grad. Accu. |   Resolution       |   Acceleration   |   Time(ms/step)  |   FPS (img/s)|
|---------------|---------------|--------------|:-------------------:|:------------------:|:----------------:|:----------------:|:----------------:|
| SDXL-Base     |    D910*x1-MS2.2.10     |    lora   |      1x1             |     1024x1024         | Graph,DS   |       539.77         |    1.85       |
| SDXL-Base     |    D910*x1-MS2.2.10     |    lora   |      1x1             |     1024x1024         | Graph,FA, DS  |       524.38          |    1.91   |
> Context: {Ascend chip}-{number of NPUs}-{mindspore version}.
>
> Acceleration: DS: data sink mode. FA: flash attention.
>
>FPS: images per second during training. average training time (s/step) = batch_size / FPS
