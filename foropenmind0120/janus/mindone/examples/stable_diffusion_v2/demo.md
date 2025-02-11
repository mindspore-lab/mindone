# 1. Stable Diffusion 2.0

## 1.1 Inference

Step 1. Download the [SD2.0 checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt) and put it under `models/` folder

Step 2. Run `text_to_image.py` to generate images for the prompt of your interest.


```shell
# Stable Diffusion 2.0 Inference
!python text_to_image.py --prompt 'A cute wolf in winter forest' -v 2.0
```

    [2023-07-04 10:54:58] INFO: Loading model from models/sd_v2_base-57526ee4.ckpt
    [2023-07-04 10:55:40] INFO: Net params not loaded: ['first_stage_model.encoder.down.3.downsample.conv.weight', 'first_stage_model.encoder.down.3.downsample.conv.bias', 'first_stage_model.decoder.up.0.upsample.conv.weight', 'first_stage_model.decoder.up.0.upsample.conv.bias']
    [2023-07-04 10:55:40] INFO: Key Settings:
    ==================================================
    MindSpore mode[GRAPH(0)/PYNATIVE(1)]: 0
    Distributed mode: False
    Number of input prompts: 1
    Number of trials for each prompt: 2
    Number of samples in each trial: 8
    Model: StableDiffusion v2.0
    Precision: Float16
    Pretrained ckpt path: models/sd_v2_base-57526ee4.ckpt
    Lora ckpt path: None
    Sampler: plms
    Sampling steps: 50
    ==================================================
    [2023-07-04 10:55:40] INFO: Running...
    [2023-07-04 10:55:40] INFO: [1/1] Sampling for prompt(s): A cute wolf in winter forest
    50it [02:21,  2.82s/it]
    [2023-07-04 10:58:58] INFO: Batch generated (8/16 imgs), time cost for this trial: 198.201s
    50it [00:23,  2.14it/s]
    [2023-07-04 10:59:23] INFO: Batch generated (16/16 imgs), time cost for this trial: 24.985s
    [2023-07-04 10:59:23] INFO: Done! All generated images are saved in: output/samples
    Enjoy.


> Note: The SD2.0 checkpoint does NOT well support Chinese prompts. If you prefer to use Chinese prompts, please refer to Section 2.1.

<details>

  <summary>Long Prompts Support</summary>

  By Default, SD V2(1.5) only supports the token sequence no longer than 77. For those sequences longer than 77, they will be truncated to 77, which can cause information loss.

  To avoid information loss for long text prompts, we can divide one long tokens sequence (N>77) into several shorter sub-sequences (N<=77) to bypass the constraint of context length of the text encoders. This feature is supported by `args.support_long_prompts` in `text_to_image.py`.

  When running inference with `text_to_image.py`, you can set the arguments as below.

  ```bash
  python text_to_image.py \
  ...  \  # other arguments configurations
  --support_long_prompts True \  # allow long text prompts
  ```
</details>

```shell
# The generated images are saved in `output/samples` folder by default
!ls output/samples
```


```python
# let's see what it makes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('output/samples/00012.png')
imgplot = plt.imshow(img)
plt.show()
```



![output_4_0](https://github.com/SamitHuang/mindone/assets/8156835/65fc481a-cbc9-4000-b1b7-94875ef76e43)



## 1.2 SD2.0 Finetune (Vanilla)

Step 1. Download the [SD2.0 checkpoint](https://download.mindspore.cn/toolkits/minddiffusion/stablediffusion/stablediffusionv2_512.ckpt) and put it under `models/` folder

Step 2. Prepare your image-text pair data (referring to README.md) and change `data_path` to your local path.

Step 3. Run the training script as follows


```shell
# After preparing the checkpoint and image-text pair data, run the follow script to finetune SD2.0 on a single NPU

python train_text_to_image.py \
    --train_config "configs/train/train_config_vanilla_v2.yaml" \
    --data_path "datasets/pokemon_blip/train" \
    --output_path "output/finetune_pokemon/txt2img" \
    --pretrained_model_path "models/sd_v2_base-57526ee4.ckpt"
```

    [2023-07-04 11:02:49] INFO: Total number of training samples: 833
    [2023-07-04 11:02:49] INFO: Num batches for rank 0: 277
    [2023-07-04 11:03:12] INFO: Loading pretrained_ckpt models/sd_v2_base-57526ee4.ckpt
    [2023-07-04 11:03:34] INFO: Params not load: ['first_stage_model.encoder.down.3.downsample.conv.weight', 'first_stage_model.encoder.down.3.downsample.conv.bias', 'first_stage_model.decoder.up.0.upsample.conv.weight', 'first_stage_model.decoder.up.0.upsample.conv.bias']
    [2023-07-04 11:03:44] INFO: Key Settings:
    ==================================================
    MindSpore mode[GRAPH(0)/PYNATIVE(1)]: 0
    Distributed mode: False
    Data path: /home/yx/datasets/diffusion/pokemon
    Model: StableDiffusion v2.0
    Precision: Float16
    Use LoRA: False
    LoRA rank: 4
    Learning rate: 1e-05
    Batch size: 3
    Grad accumulation steps: 1
    Num epochs: 20
    Grad clipping: False
    Max grad norm: 1.0
    EMA: False
    ==================================================
    [2023-07-04 11:03:44] INFO: Start training...
    epoch: 1 step: 1, loss is 0.16235122
    epoch: 1 step: 2, loss is 0.11218587
    epoch: 1 step: 3, loss is 0.06332338
    epoch: 1 step: 4, loss is 0.059139673
    epoch: 1 step: 5, loss is 0.13915305
    epoch: 1 step: 6, loss is 0.014662254
    epoch: 1 step: 7, loss is 0.054861214
    epoch: 1 step: 8, loss is 0.04518071
    ...
    epoch: 20 step: 276, loss is 0.021809997
    epoch: 20 step: 277, loss is 0.11882311
    Train epoch time: 130989.351 ms, per step time: 472.886 ms
    Checkpoint saved in  output/finetune_pokemon/txt2img/ckpt/rank_0/sd-20.ckpt


## 1.3 LoRA Finetune

For detailed illustration, please refer to [Use LoRA for Stable Diffusion Finetune](lora_finetune.md)

```shell
python train_text_to_image.py \
    --train_config "configs/train/train_config_lora_v2.yaml" \
    --data_path "datasets/pokemon_blip/train" \
    --output_path "output/lora_pokemon/txt2img" \
    --pretrained_model_path "models/sd_v2_base-57526ee4.ckpt"
```

    [2023-07-04 11:57:45] INFO: Total number of training samples: 833
    [2023-07-04 11:57:45] INFO: Num batches for rank 0: 208
    [2023-07-04 11:58:02] INFO: Loading pretrained model from models/sd_v2_base-57526ee4.ckpt
    [2023-07-04 11:58:18] INFO: Params not load: ['first_stage_model.encoder.down.3.downsample.conv.weight', 'first_stage_model.encoder.down.3.downsample.conv.bias', 'first_stage_model.decoder.up.0.upsample.conv.weight', 'first_stage_model.decoder.up.0.upsample.conv.bias']
    [2023-07-04 11:58:20] INFO: LoRA enabled. Number of injected params: 256
    [2023-07-04 11:58:20] INFO: Key Settings:
    ==================================================
    MindSpore mode[GRAPH(0)/PYNATIVE(1)]: 0
    Distributed mode: False
    Data path: /home/yx/datasets/diffusion/pokemon
    Model: StableDiffusion v2.0
    Precision: Float16
    Use LoRA: True
    LoRA rank: 128
    Learning rate: 0.0001
    Batch size: 4
    Grad accumulation steps: 1
    Num epochs: 72
    Grad clipping: True
    Max grad norm: 1.0
    EMA: True
    ==================================================
    [2023-07-04 11:58:20] INFO: Start training...
    epoch: 1 step: 1, loss is 0.04827215
    epoch: 1 step: 2, loss is 0.040287398
    epoch: 1 step: 3, loss is 0.033867083
    epoch: 1 step: 4, loss is 0.016457044
    epoch: 1 step: 5, loss is 0.08254205
    epoch: 1 step: 6, loss is 0.049230024
    epoch: 1 step: 7, loss is 0.04292356
    epoch: 1 step: 8, loss is 0.10743075
    ...
    epoch: 72 step: 207, loss is 0.007903519
    epoch: 72 step: 208, loss is 0.03977389
    Train epoch time: 105981.696 ms, per step time: 509.527 ms
    Checkpoint saved in  output/lora_pokemon/txt2img/ckpt/rank_0/sd-72.ckpt

## 1.4 Inference with Finetuned Model

```shell
!python text_to_image.py --prompt "A drawing of a fox with a red tail" --use_lora True --lora_rank 128 --lora_ckpt_path output/lora_pokemon/txt2img/ckpt/rank_0/sd-72.ckpt
```

    [2023-07-05 09:18:36] INFO: Loading pretrained model from models/sd_v2_base-57526ee4.ckpt
    [2023-07-05 09:19:09] INFO: LoRA enabled. Number of injected params: 256
    [2023-07-05 09:19:09] INFO: Loading LoRA params from output/lora_pokemon/txt2img/ckpt/rank_0/sd-72.ckpt
    [2023-07-05 09:19:09] INFO: Key Settings:
    ==================================================
    MindSpore mode[GRAPH(0)/PYNATIVE(1)]: 0
    Distributed mode: False
    Number of input prompts: 1
    Number of trials for each prompt: 2
    Number of samples in each trial: 8
    Model: StableDiffusion v2.0
    Precision: Float16
    Pretrained ckpt path: models/sd_v2_base-57526ee4.ckpt
    Lora ckpt path: output/lora_pokemon/txt2img/ckpt/rank_0/sd-72.ckpt
    Sampler: plms
    Sampling steps: 50
    Uncondition guidance scale: 9.0
    ==================================================
    [2023-07-05 09:19:09] INFO: Running...
    [2023-07-05 09:19:09] INFO: [1/1] Generating images for prompt(s):
    A drawing of a fox with a red tail
    50it [01:59,  2.38s/it]
    [2023-07-05 09:21:58] INFO: 8/16 images generated, time cost for current trial: 169.287s
    50it [00:24,  2.05it/s]
    [2023-07-05 09:22:24] INFO: 16/16 images generated, time cost for current trial: 25.784s
    [2023-07-05 09:22:24] INFO: Done! All generated images are saved in: output/samples
    Enjoy.


For example results, please refer to [here](lora_finetune.md#inference).

# 2. Stable Diffusion 1.x (Chinese)

## 2.1 Inference

Step 1. Download the [SD1.x checkpoint](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) (credit to WuKongHuaHua) and put it under `models/` folder

Step 2. Run `text_to_image.py` to generate images for the prompt of your interest and specify the `-v` arg.



```shell
# Stable Diffusion 1.x Inference
!python text_to_image.py --prompt '雪中之狼' -v 1.x
```


```shell
# The generated images are saved in `output/samples` folder by default
!ls output/samples
```


```python
img = mpimg.imread('output/samples/00030.png')
imgplot = plt.imshow(img)
plt.show()
```

## 2.2 SD1.x Finetune (Vanilla)

Step 1.Download the [SD1.x checkpoint](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) (credit to WuKongHuaHua) and put it under `models/` folder

Step 2. Prepare your image-text pair data (referring to README.md) and change `data_path` to your local path.

Step 3. Run the training script as follows


```shell
# After preparing the checkpoint and image-text pair data, run the follow script to finetune SD1.5 on a single NPU

python train_text_to_image.py \
    --train_config "configs/train/train_config_vanilla_v1_chinese.yaml" \
    --data_path datasets/pokemon_cn/train \
    --output_path "output/txt2img" \
    --pretrained_model_path "models/wukong-huahua-ms.ckpt"
```
