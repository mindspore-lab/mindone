# Lora Finetune

We provide the script `train.py` for lora finetuning of sdxl.

## Requirements

| mindspore      | ascend driver | firmware    | cann toolkit/kernel |
| :--------------: | :------------: |:----------: |:------------------: |
| 2.2.10～2.2.12 | 23.0.3        | 7.1.0.5.220 | 7.0.0.beta1         |

## Pretrained models

Download the official pre-train weights from huggingface, convert the weights from `.safetensors` format to Mindspore `.ckpt` format, and put them in `./checkpoints/` folder. Please refer to SDXL [weight_convertion](./preparation.md#convert-pretrained-checkpoint) for detailed steps.

The scripts automatically download the clip tokenizer, but if you have network issues with it, please manually download [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) from huggingface and change `version: openai/clip-vit-large-patch14` in `configs/inference/sd_xl_base.yaml` to `version: your_path/to/clip-vit-large-patch14`.

## lora finetune

```shell
# sdxl-base lora fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_base_finetune_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \
  --gradient_accumulation_steps 4 \
```

## Inference, run lora(unmerge weight) without streamlit on Ascend

```shell
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_lora.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt,SDXL-base-1.0_2000_lora.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --device_target Ascend
```

## Performance

Experiments are tested on ascend 910* with mindspore 2.2.10 graph mode.

| Model Name      | Card | bs * grad accu. |   Resolution       |   acceleration   |   Time(ms/step)  |   FPS (img/s)|
|---------------|:------------------:|:----------------:|:----------------:|:----------------:|------------------|:----------------:|
| SDXL-Base     |      1            |      1x1             |     1024x1024         | DS           |       539.77         |    1.85       |
| SDXL-Base     |      1            |      1x1             |     1024x1024         | FA, DS |       524.38          |    1.91   |
> Acceleration: DS: data sink mode. FA: flash attention.
>
>FPS: images per second during training. average training time (s/step) = batch_size / FPS
