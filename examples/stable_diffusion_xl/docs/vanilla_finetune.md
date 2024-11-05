# Vanilla Finetune

We provide the script `train.py` for full parameter training of sdxl.

## Requirements

| mindspore      | ascend driver | firmware    | cann toolkit/kernel |
|:--------------:| :------------:| :----------:| :------------------:|
|  2.3.1    | 24.1.RC2      | 7.3.0.1.231 | 8.0.RC2.beta1        |

## Pretrained models

Please follow SDXL [weight convertion](./preparation.md#convert-pretrained-checkpoint) for detailed steps and put the pretrained weight to `./checkpoints/`.

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
    --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
    --data_path /PATH TO/YOUR DATASET/ \
    --max_device_memory "59GB" \
    --param_fp16 True \
    --is_parallel True
```

### 3. data sink mode
Add `--data_sink True` and `--sink_size 100` to the command line to enable data sink mode. For example,
```shell
msrun --worker_num=8 \
  --local_worker_num=8 \
  --bind_core=True \
  --join=True \
  --log_dir=8p_bs1_fa_sink_sdxl_base_log \
  python train.py \
    --config configs/training/sd_xl_base_finetune_910b_fa.yaml \
    --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
    --data_path /PATH TO/YOUR DATASET/ \
    --save_path_with_time False \
    --max_device_memory "59GB" \
    --data_sink True \
    --sink_size 100 \
    --param_fp16 True \
    --is_parallel True
```

## Performance

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

| model name | cards |  batch size  |resolution| precision |  flash attn  | sink |jit level| graph compile | s/step |  img/s  |
| :--------: | :---: |:-----------: | :------:|  :--: | :------: | :--: | :--: | :-------: | :---: | :---: |
| SDXL-Base  | 1     |1             | 1024x1024  |     fp32   | OFF  | OFF  |O2|    20~25 mins   | 0.72   | 1.38  |
| SDXL-Base  | 8     |1             | 1024x1024  |     fp16    | OFF  | OFF  |O2|   30~35 mins   |  0.88   | 9.09  |
| SDXL-Base  | 8     |1             | 1024x1024  |    fp16    |  ON  |  ON  |O2|    30~35 mins   | 0.53   | 15.09 |
| SDXL-Base  | 8     |2             | 1024x1024  |     fp16   |  ON  |  ON  |O2|    30~35 mins   | 0.71   | 22.54 |
| SDXL-Base  | 8     |4             | 1024x1024  |     fp16   |  ON  |  ON  |O2|    30~38 mins   | 1.07   | 29.91 |
> precision here means the amp, which is controled by the arg `--param_fp16` of the script.
