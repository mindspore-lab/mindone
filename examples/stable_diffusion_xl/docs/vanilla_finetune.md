# Vanilla Finetune

We provide the script `train.py` for full parameter training of sdxl.

## Requirements

| mindspore      | ascend driver | firmware    | cann toolkit/kernel |
|:--------------:| :------------:| :----------:| :------------------:|
|  2.3.1    | 24.1.RC2      | 7.3.0.1.231 | 8.0.RC2.beta1        |

## Pretrained models

Please follow SDXL [weight_convertion](./preparation.md#convert-pretrained-checkpoint) for detailed steps and put the pretrained weight to `./checkpoints/`.

The scripts automatically download the clip tokenizer. If you have network issues with it, [FAQ Qestion 5](./faq_cn.md#5-连接不上huggingface-报错-cant-load-tokenizer-for-openaiclip-vit-large-patch14) helps.

## Datasets preparation
See [dataset preparation](./preparation.md#dataset-preparation-for-fine-tuning-optional). Csv, webdataset or wids format are supported.

## Config guide

Please refer to [config guide](./config_guide.md) for hyper-parameters setting.

> [!WARNING]
> It is not recommended to turn on `--param_fp16`, which will force weight conversion to `fp16` and may lead to unstable training.

## Finetuning

### 1. vanilla fine-tune
```shell
# sdxl-base fine-tune, standalone
python train.py \
  --config configs/training/sd_xl_base_finetune_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/
```

### 2. vanilla fine-tune for distribute

Distributed training with `msrun` command,
```shell
# sdxl-base fine-tune with 8p
msrun --worker_num=8 \
  --local_worker_num=8 \
  --bind_core=True \
  --join=True \
  --log_dir=8p_bs1_sdxl_base_log \
  python train.py \
    --config configs/training/sd_xl_base_finetune_910b.yaml \
    --weight "" \
    --data_path /PATH TO/YOUR DATASET/ \
    --max_device_memory "59GB" \
    --param_fp16 True \
    --is_parallel True
```
or prepare hccl [rank table](../tools/rank_table_generation/README.md) file for single/multi-server(s),

```shell
# sdxl-base fine-tune with 8p
bash scripts/run_distribute_vanilla_ft_910b.sh /path_to/hccl_8p.json 0 8 8 /path_to/dataset/

# sdxl-base fine-tune with 16p on Ascend
bash scripts/run_distribute_vanilla_ft_910b.sh /path_to/hccl_16p.json 0 8 16 /path_to/dataset/  # run on server 1
bash scripts/run_distribute_vanilla_ft_910b.sh /path_to/hccl_16p.json 8 16 16 /path_to/dataset/ # run on server 2
```

### 3. vanilla fine-tune for cache latent and text-embedding

#### 3.1. cache dataset

```shell
# standalone
python train.py \
  --task cache \
  --config configs/training/sd_xl_base_finetune_910b_fa.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path $DATASET_PATH \
  --save_path_with_time False \
  --cache_latent True \
  --cache_text_embedding True \
  --cache_path ./cache_data

# 8p
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


## Performance

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

| model name | cards | image size | graph compile |  local bs  | amp fp16 |  fa  | cache | sink | s/step |  fps  |
| :--------: | :---: | :--------: | :-----------: | :--: | :------: | :--: | :---: | :--: | :-------: | :---: |
| SDXL-Base  | 1  | 1024x1024  |  20~25 mins   | 1  |    off   | off  |  off  | off  |   0.72s   | 1.38  |
| SDXL-Base  | 8  | 1024x1024  |  30~35 mins   | 1  |    on    | off  |  off  | off  |   0.88s   | 9.09  |
| SDXL-Base  | 8  | 1024x1024  |  30~35 mins   | 1  |    on    |  on  |  on   |  on  |   0.53s   | 15.09 |
| SDXL-Base  | 8  | 1024x1024  |  30~35 mins   | 2  |    on    |  on  |  on   |  on  |   0.71s   | 22.54 |
| SDXL-Base  | 8  | 1024x1024  |  30~38 mins   | 4  |    on    |  on  |  on   |  on  |   1.07s   | 29.91 |
> fa: flash attention. fps: images per second during training, average training time (s/step) = batch_size / fps
