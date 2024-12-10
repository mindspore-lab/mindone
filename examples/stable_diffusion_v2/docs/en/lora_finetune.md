# LoRA for Stable Diffusion Fine-tuning
> [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Introduction
LoRA was first introduced by Microsoft for efficient large-language model finetuning in 2021. LoRA freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks, as shown in Figure. 1.

<p align="center">
  <img src="https://github.com/SamitHuang/mindone/assets/8156835/2c781c73-0eac-4cb6-92e8-5a40c9738e9c" width=250 />
</p>
<p align="center">
  <em> Figure 1. Illustration of a LoRA Injected Module [<a href="#references">1</a>] </em>
</p>

LoRA allows adapting pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:

- Previous pretrained weights are kept frozen so that model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114).
- Rank-decomposition matrices have significantly fewer parameters than the original model.
- LoRA finetuning is more efficient (approximately 2 times faster) than other finetuning approaches such as DreamBooth and Texture Inversion.
- Support merging different LoRAs together

LoRA was extended to finetune diffusion models by [cloneofsimo](https://github.com/cloneofsimo/lora).
For latent diffusion models, LoRA is usually applied to the CrossAttention layers in UNet, and can also be applied to the Attention layers in the text encoder.


## Preparation

### Dependency

Please refer to the [Installation](../../README.md#installation) section.

### Pretrained Weights

We support LoRA fine-tuning on pretrained text-to-image models as listed in [Pretrained Text-to-Image Models](../../README.md#preparing-pretrained-weights).
Please download one of the pretrained checkpoint from the table and put it in `models` folder.

### Text-image Dataset Preparation

Please refer to the [Dataset Preparation](../../README.md#dataset-preparation) section.

## LoRA Fine-tuning

After preparing the pretrained weight and fine-tuning dataset, you can use the `train_text_to_image.py` script and set argument `use_lora=True` for LoRA fine-tuning.

To run LoRA fine-tuning, please specify the `train_config`, `data_path`, and `pretrained_model_path` arguments according to the model and data you want to fine-tune with, then execute

```shell
python train_text_to_image.py \
    --train_config {path to a pre-defined training config yaml} \
    --data_path {path to training data directory} \
    --output_path {path to output directory} \
    --pretrained_model_path {path to pretrained checkpoint file}
```

The training configurations are specified via the `train_config` argument, including model architecture and the training hyper-parameters such as `lora_rank`.

The key arguments, which can be changed in the YAML file of `train_config` or CLI, are as follows:
- `use_lora`: whether fine-tune with LoRA
- `lora_rank`: the rank of the low-rank matrices in lora params, default: 4.
- `lora_ft_text_encoder`: whether fine-tune the text encoder with LoRA, default: False.
- `lora_fp16`:  whether compute LoRA in float16, default: True
- `model_config`: path to the model architecture configuration file.
- `pretrained_model_path`: path to the pretrained model weight

For more argument illustration, please run `python train_text_to_image.py -h`.

The trained LoRA checkpoints will be saved in `{output_path}/ckpt`, which are small since only LoRA parameters are saved .

### Example 1: Fine-tuning SD1.5 with LoRA on Pokemon Dataset

After downloading [sd_v1.5-d0ab7146.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt) to `models` folder, please run

```shell
python train_text_to_image.py \
    --train_config configs/train/train_config_lora_v1.yaml \
    --data_path datasets/pokemon_blip/train \
    --output_path output/lora_pokemon \
    --pretrained_model_path models/sd_v1.5-d0ab7146.ckpt
```

The trained LoRA checkpoints will be saved in `output/lora_pokemon/ckpt`.

For fine-tuning other SD1.x checkpoints, please change `pretrained_model_path` accordingly.

### Example 2: Fine-tuning SD2.1 with LoRA on Chinese Dataset

After downloading [sd_v2-1_base-7c8d09ce.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2-1_base-7c8d09ce.ckpt) to `models` folder, please run

```shell
python train_text_to_image.py \
    --train_config configs/train/train_config_lora_v2.yaml \
    --data_path datasets/chinese_art_blip/train \
    --output_path output/lora_chinese_art \
    --pretrained_model_path models/sd_v2-1_base-7c8d09ce.ckpt
```

The trained LoRA checkpoints will be saved in `output/lora_chinese_art/ckpt`.

For fine-tuning other SD2.x checkpoints, please change `pretrained_model_path` accordingly.


## Inference

To perform text-to-image generation with the fine-tuned lora checkpoint, please run

```shell
python text_to_image.py \
        --prompt "A drawing of a fox with a red tail" \
        --use_lora True \
        --lora_ckpt_path {path/to/lora_checkpoint_after_finetune} \
        --version {Stable diffusion version}
```

Please update `lora_ckpt_path` and `version`according to your fine-tuning settings.

Here are the example results.

<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/27ac4537-b407-4196-a17d-8a8351d150ec" width="160" height="160" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/791ce730-4fee-457d-8272-9f88febdd854" width="160" height="160" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/6839afa0-6fc3-4320-942d-7cd7c2dcb8ab" width="160" height="160" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/f8ea3071-f279-47d3-9fc0-08fe6d9009e8" width="160" height="160" />
</div>
<p align="center">
  <em> Images generated by Stable Diffusion 2.0 fine-tuned on pokemon-blip dataset using LoRA </em>
</p>

<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/dfb1bbe2-185e-4367-a1a8-1c1716a693d4" width="160" height="160" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/995fab36-7015-4176-8102-a79080f5b0c7" width="160" height="160" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/4279fc0d-035b-4e77-a50a-9deac58bab84" width="160" height="160" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/280a4ea8-6988-4e4a-83a7-0944f000ac76" width="160" height="160" />
</div>
<p align="center">
  <em> Images generated by Stable Diffusion 2.0 fine-tuned on chinese-art-blip dataset using LoRA </em>
</p>

## Evaluation

We will evaluate the finetuned model on the split test set in `pokemon_blip.zip` and `chinese_art_blip.zip`.

Let us run text-to-image generation conditioned on the prompts in test set then evaluate the quality of the generated images by the following steps.

1. Before running, please modify the following arguments to your local path:

* `--data_path=/path/to/prompts.txt`
* `--output_path=/path/to/save/output_data`
* `--lora_ckpt_path=/path/to/lora_checkpoint`

`prompts.txt` is a file which contains multiple prompts, and each line is the caption for a real image in test set, for example

```text
a drawing of a spider on a white background
a drawing of a pokemon with blue eyes
a drawing of a pokemon pokemon with its mouth open
...
```

2. Run multiple-prompt inference on the test set

```shell
python text_to_image.py \
    --version "2.0" \
    --config "configs/v2-inference.yaml" \
    --output_path "output/lora_pokemon_infer" \
    --n_iter 1 \
    --n_samples 2 \
    --scale 9.0 \
    --W 512 \
    --H 512 \
    --use_lora True \
    --lora_ft_text_encoder False \
    --lora_ckpt_path "output/lora_pokemon/txt2img/ckpt/rank_0/sd-72.ckpt" \
    --dpm_solver \
    --sampling_steps 20 \
    --data_path datasets/pokemon_blip/test/prompts.txt
```

The generated images will be saved in the `{output_path}/samples` folder.

Note that the following hyper-param configuration will affect the generation and evaluation results.

- sampler: the diffusion sampler
- sampling_steps: the sampling steps
- scale: unconditional guidance scale

For more details, please run `python text_to_image.py -h`.

3. Evaluate the generated images

```shell
python eval/eval_fid.py --real_dir {path/to/test_images} --gen_dir {path/to/generated_images}
python eval/eval_clip_score.py --image_path_or_dir {path/to/generated_images} --prompt_or_path {path/to/prompts_file} --ckpt_path {path/to/checkpoint}
```

For details, please refer to the guideline [Diffusion Evaluation](tools/eval/README.md).
## Notes

- Apply LoRA to Text-encoder

By default, LoRA fintuning is only applied to UNet. To finetune the text encoder with LoRA as well, please pass `--lora_ft_text_encoder=True` to the finetuning script (`train_text_to_image.py`) and inference script (`text_to_image.py`).


## Reference
[1] [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
