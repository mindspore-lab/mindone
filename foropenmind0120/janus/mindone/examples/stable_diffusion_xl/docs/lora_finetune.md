# LoRA Finetune

We provide the script `train.py` for [LoRA](https://arxiv.org/abs/2106.09685) finetune of sdxl.

## Requirements

| mindspore      | ascend driver | firmware    | cann toolkit/kernel |
| :--------------:| :------------: |:----------: |:------------------: |
|  2.3.1    | 24.1.RC2      | 7.3.0.1.231 | 8.0.RC2.beta1        |

## Pretrained models

Please follow SDXL [weight conversion](./preparation.md#convert-pretrained-checkpoint) for detailed steps and put the pre-trained weight to `./checkpoints/`.

The scripts automatically download the clip tokenizer. If you have network issues with it, [FAQ Qestion 5](./faq_cn.md#5-连接不上huggingface-报错-cant-load-tokenizer-for-openaiclip-vit-large-patch14) helps.

## Datasets preparation

See [dataset preparation](./preparation.md#general-text-image-datasets) to prepare a general text-image dataset for LoRA finetune.

## Finetuning

```shell
# sdxl-base lora fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_base_finetune_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \
  --gradient_accumulation_steps 4
```

## Inference
Run lora (unmerge weight) without streamlit,

```shell
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_lora.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt,SDXL-base-1.0_2000_lora.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --device_target Ascend
```

## Performance

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

| model name| cards  |batch size  |    resolution    | grad accu        |   sink   | flash attn |jit level| graph compile |  ms/step  |   img/s |
|:---------:|:------:|:----------:|:--------:       |:----------------:|:--------:|:--:|:----------:|:------:|------------------|:----------------:|
| SDXL-Base | 1      | 1          | 1024x1024       | 1                | ON  |OFF  |O2| 20~25mins    | 553.47 | 1.80 |
| SDXL-Base | 1      | 1          | 1024x1024       | 1                | ON  |ON   |O2| 20~25mins    | 446.74 | 2.24 |
> grad accu: gradient accumulation steps.
