# Vanilla Finetune

### Introduction

We provide the script `train.py` for full parameter training of sdxl.

> Note: If you have network issues on downloading clip tokenizer, please manually download [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) from huggingface and change `version: openai/clip-vit-large-patch14` in `configs/inference/sd_xl_base.yaml` to `version: your_path/to/clip-vit-large-patch14`

### Pretrained models

Download the official pre-train weights from huggingface, convert the weights from `.safetensors` format to Mindspore `.ckpt` format, and put them to `./checkpoints/` folder. Please refer to SDXL [weight_convertion.md](./weight_convertion.md) for detailed steps.

### hyper-parameters setting

[hyper_parameters.md](./hyper_parameters.md)

> ⚠️: It is not recommended to turn on `--param_fp16`, that will force to convert the weight to `fp16` and may lead to unstable training.

### 1. vanilla fine-tune

```shell
# sdxl-base fine-tune
python train.py \
  --config configs/training/sd_xl_base_finetune_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/
```

### 2. vanilla fine-tune for distribute

Prepare hccl [rank_table](./tools/rank_table_generation/README.md) file for single/multi-server(s).

```shell
# sdxl-base fine-tune with 8p
mpirun --allow-run-as-root -n 8 python train.py \
  --config configs/training/sd_xl_base_finetune_910b.yaml \
  --weight "" \
  --data_path /PATH TO/YOUR DATASET/ \
  --max_device_memory "59GB" \
  --param_fp16 True \
  --is_parallel True
or
bash scripts/run_distribute_vanilla_ft_910b.sh /path_to/hccl_8p.json 0 8 8 /path_to/dataset/

# sdxl-base fine-tune with 16p on Ascend
bash scripts/run_distribute_vanilla_ft_910b.sh /path_to/hccl_16p.json 0 8 16 /path_to/dataset/  # run on server 1
bash scripts/run_distribute_vanilla_ft_910b.sh /path_to/hccl_16p.json 8 16 16 /path_to/dataset/ # run on server 2
```

### 3. vanilla fine-tune for cache latent and text-embedding

#### 3.1. cache dataset

```shell
python train.py \
  --task cache \
  --config configs/training/sd_xl_base_finetune_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path $DATASET_PATH \
  --save_path_with_time False \
  --cache_latent True \
  --cache_text_embedding True \
  --cache_path ./cache_data
or
bash scripts/cache_data.sh /path_to/hccl_8p.json 0 8 8 /path_to_dataset/ /path_to_cache/ # cache data
```

#### 3.2. train with cache data

```shell
# sdxl-base fine-tune with cache on Ascend
bash scripts/run_distribute_vanilla_ft_910b_cache.sh /path_to/hccl_8p.json 0 8 8 /path_to_dataset/  # run on server 1
```

#### 3.3. merge weight and infer

It is necessary to merge trained Unet weight and pre-trained weight before inference, because only the weight of UNet are saved when use cache.

```shell
# merge weight
python tools/weight_merge/merge_weight.py \
  --base_weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --additional_weights unet.ckpt \
  --save_path merged_weight.ckpt

# sdxl-base run infer
python demo/sampling_without_streamlit.py \
  --weight /path_to/merged_weight.ckpt \
  --prompt "your prompt"
```

### 4. resume vanilla fine-tune

```shell
# resume sdxl-base fine-tune from specified training step
python train.py \
  --config configs/training/sd_xl_base_finetune_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \
  --optimizer_weight /PATH/TO/SAVED/OPTIMIZER/WEIGHT \
  --resume_step 1000
```


### 5. Support List

[MindSpore 2.2.11](https://www.mindspore.cn/versions#2.2.11)

<div align="center">

| model name | device | cards | imagesize | graph compile | bs  | amp fp16 | fa  | cache | sink | step time |  fps  |
|:----------:|:------:|:-----:|:---------:|:-------------:|:---:|:--------:|:---:|:-----:|:----:|:---------:|:-----:|
| SDXL-Base  |  910*  | 1*8p  | 1024x1024 |  30~38 mins   | 1*8 |    on    | off |  off  | off  |   1.10s   | 7.27  |
| SDXL-Base  |  910*  | 1*8p  | 1024x1024 |  30~38 mins   | 1*8 |    on    | on  |  on   |  on  |   0.74s   | 10.81 |
| SDXL-Base  |  910*  | 1*8p  | 1024x1024 |  30~38 mins   | 2*8 |    on    | on  |  on   |  on  |   0.87s   | 18.39 |
| SDXL-Base  |  910*  | 1*8p  | 1024x1024 |  30~38 mins   | 4*8 |    on    | on  |  on   |  on  |   1.38s   | 23.18 |
| SDXL-Base  |  910*  | 1*8p  | 1024x1024 |  30~38 mins   | 6*8 |    on    | on  |  on   |  on  |   1.96s   | 24.48 |
| SDXL-Base  |  910*  | 1*8p  | 1024x1024 |  30~38 mins   | 8*8 |    on    | on  |  on   |  on  |   2.51s   | 25.52 |

</div>
<br>
