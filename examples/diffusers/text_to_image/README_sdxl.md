# Stable Diffusion XL text-to-image fine-tuning

The `train_text_to_image_sdxl.py` script shows how to fine-tune Stable Diffusion XL (SDXL) on your own dataset.

🚨 This script is experimental. The script fine-tunes the whole model and often times the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyperparameters to get the best result on your dataset. 🚨

## Running locally with MindSpore

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

The training script is compute-intensive and only runs on an Ascend 910*. Please run the scripts with CANN version ([CANN 8.0.RC2.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1)) and MindSpore version ([MS 2.3.0](https://www.mindspore.cn/versions#2.3.0)). You can use
`cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` check the CANN version and you can see the specific version number [7.3.0.1.231:8.0.RC2]. If you have a custom installation path for CANN, find the `version.cfg` in your own CANN installation path to verify the version.

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/mindspore-lab/mindone
cd mindone
pip install -e .
```

Then cd in the `examples/diffusers/text_to_image` folder and run
```bash
pip install -r requirements_sdxl.txt
```

### Training

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="YaYaB/onepiece-blip-captions"

python train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --max_train_steps=10000 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --validation_prompt="a man in a green coat holding two swords" --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="sdxl-onepiece-model-$(date +%Y%m%d%H%M%S)"
```

For parallel training, use `msrun` and along with `--distributed`:

```shell
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="YaYaB/onepiece-blip-captions"

msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
    train_text_to_image_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --pretrained_vae_model_name_or_path=$VAE_NAME \
    --dataset_name=$DATASET_NAME \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 --center_crop --random_flip \
    --proportion_empty_prompts=0.2 \
    --train_batch_size=1 \
    --max_train_steps=10000 \
    --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --validation_prompt="a man in a green coat holding two swords" --validation_epochs 5 \
    --checkpointing_steps=5000 \
    --distributed \
    --output_dir="sdxl-onepiece-model-$(date +%Y%m%d%H%M%S)"
```

**Notes**:

*  The `train_text_to_image_sdxl.py` script pre-computes text embeddings and the VAE encodings and keeps them in memory. While for smaller datasets like [`lambdalabs/pokemon-blip-captions`](https://hf.co/datasets/lambdalabs/pokemon-blip-captions), it might not be a problem, it can definitely lead to memory problems when the script is used on a larger dataset. For those purposes, you would want to serialize these pre-computed representations to disk separately and load them during the fine-tuning process. Refer to [this PR](https://github.com/huggingface/diffusers/pull/4505) for a more in-depth discussion.
* The training command shown above performs intermediate quality validation in between the training epochs. `--report_to`, `--validation_prompt`, and `--validation_epochs` are the relevant CLI arguments here.
* SDXL's VAE is known to suffer from numerical instability issues. This is why we also expose a CLI argument namely `--pretrained_vae_model_name_or_path` that lets you specify the location of a better VAE (such as [this one](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)).

### Performance

For the above training example, we record the training speed as follows.

| Method  | NPUs | Global <br/>Batch size | Resolution   | Precision | Graph Compile | Speed <br/>(s/step) | FPS <br/>(img/s) |
|---------|------|------------------------|--------------|-----------|---------------|---------------------|------------------|
| vanilla | 1    | 1*1                    | 512x512      | FP16      | 1~5 mins      | 0.720               | 1.39             |
| vanilla | 8    | 1*8                    | 512x512      | FP16      | 1~5 mins      | 1.148               | 6.97             |

### Inference

```python
from mindone.diffusers import DiffusionPipeline
import mindspore

model_path = "stabilityai/stable-diffusion-xl-base-1.0" # <-- You can modify the model path of your training here.
pipe = DiffusionPipeline.from_pretrained(model_path, mindspore_dtype=mindspore.float16)

prompt = "The boy rides a horse in space"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)[0][0]
image.save("The-boy-rides-a-horse-in-space.png")
```

To change the pipelines scheduler, use the from_config() method to load a different scheduler's pipeline.scheduler.config into the pipeline.

```python
from mindone.diffusers import EulerAncestralDiscreteScheduler

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)[0][0]
image.save("The-boy-rides-a-horse-in-space.png")
```

Here are some images generated by inference under different Schedulers.

|                                                                   DDIMParallelScheduler <br/>(0.86s/step)                                                                    |                                                                    DDIMScheduler <br/>(0.8s/step)                                                                    |                                                                   LMSDiscreteScheduler <br/>(0.93s/step)                                                                    |                                                                   DPMSolverSinglestepScheduler <br/>(0.83s/step)                                                                    |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/diff_schedulers_infer/DDIMParallelScheduler.png?raw=true" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/diff_schedulers_infer/DDIMScheduler.png?raw=true" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/diff_schedulers_infer/LMSDiscreteScheduler.png?raw=true" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/diff_schedulers_infer/DPMSolverSinglestepScheduler.png?raw=true" width=224> |

Checkpoints only save the unet, so to run inference from a checkpoint, just load the unet.

```python
import mindspore as ms
from mindone.diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

model_path = "sdxl-onepiece-model"
unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-<N>/unet", mindspore_dtype=ms.float16)

pipe = StableDiffusionXLPipeline.from_pretrained("<initial model>", unet=unet, mindspore_dtype=ms.float16)

image = pipe(prompt="a man with a beard and a shirt")[0][0]
image.save("onepiece.png")
```

We trained 10k steps based on the OnePiece dataset. Here are some of the results of the fine-tuning.

|                                              a man in a blue suit and a green hat                                                                                                  |                                         a man with a big mouth                                                                                              |                                                                  a man with glasses on his face                                                                     |                                                                  a man with red hair and a cape                                                                     |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_infer_fa_10k/a_man_in_a_blue_suit_and_a_green_hat.png?raw=true" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_infer_fa_10k/a_man_with_a_big_mouth.png" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_infer_fa_10k/a_man_with_glasses_on_his_face.png" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_infer_fa_10k/a_man_with_red_hair_and_a_cape.png" width=224> |

## LoRA training example for Stable Diffusion XL (SDXL)

Low-Rank Adaption of Large Language Models was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*.

In a nutshell, LoRA allows adapting pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:

- Previous pretrained weights are kept frozen so that model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114).
- Rank-decomposition matrices have significantly fewer parameters than original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted toward new training images via a `scale` parameter.

[cloneofsimo](https://github.com/cloneofsimo) was the first to try out LoRA training for Stable Diffusion in the popular [lora](https://github.com/cloneofsimo/lora) GitHub repository.

### Training

First, you need to set up your development environment as is explained in the [installation section](#installing-the-dependencies). Make sure to set the `MODEL_NAME` and `DATASET_NAME` environment variables and, optionally, the `VAE_NAME` variable. Here, we will use [Stable Diffusion XL 1.0-base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and the [OnePiece dataset](https://huggingface.co/datasets/YaYaB/onepiece-blip-captions).

**___Note: It is quite useful to monitor the training progress by regularly generating sample images during training. [Weights and Biases](https://docs.wandb.ai/quickstart) is a nice solution to easily see generating images during training. All you need to do is to run `pip install wandb` before training to automatically log images.___**

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="YaYaB/onepiece-blip-captions"

python train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=1024 --center_crop --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --validation_prompt="a man in a green coat holding two swords" \
  --output_dir="sdxl-onepiece-model-lora-$(date +%Y%m%d%H%M%S)"
```

For parallel training, use `msrun` and along with `--distributed`:

```shell
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="YaYaB/onepiece-blip-captions"

msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
    train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --pretrained_vae_model_name_or_path=$VAE_NAME \
    --dataset_name=$DATASET_NAME \
    --resolution=1024 --center_crop --random_flip \
    --train_batch_size=1 \
    --num_train_epochs=2 --checkpointing_steps=500 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --seed=42 \
    --validation_prompt="a man in a green coat holding two swords" \
    --distributed \
    --output_dir="sdxl-onepiece-model-lora-$(date +%Y%m%d%H%M%S)"
```

The above command will also run inference as fine-tuning progresses and log the results to local files.

**Notes**:

* SDXL's VAE is known to suffer from numerical instability issues. This is why we also expose a CLI argument namely `--pretrained_vae_model_name_or_path` that lets you specify the location of a better VAE (such as [this one](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)).

### Performance

For the above training example, we record the training speed as follows.

| Method | NPUs | Global <br/>Batch size | Resolution   | Precision | Graph Compile | Speed <br/>(s/step) | FPS <br/>(img/s) |
|--------|------|------------------------|--------------|-----------|---------------|---------------------|-----------------|
| lora   | 1    | 1*1                    | 1024x1024    | FP16      | 15~20 min     | 0.828               | 1.21            |
| lora   | 8    | 1*8                    | 1024x1024    | FP16      | 15~20 min     | 0.907               | 8.82            |


### Finetuning the text encoder and UNet

The script also allows you to finetune the `text_encoder` along with the `unet`.

🚨 Training the text encoder requires additional memory.

Pass the `--train_text_encoder` argument to the training script to enable finetuning the `text_encoder` and `unet`:

```bash
python train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=1024 --center_crop --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --validation_prompt="a man in a green coat holding two swords" \
  --train_text_encoder \
  --output_dir="sdxl-onepiece-model-lora-$(date +%Y%m%d%H%M%S)"
```

For parallel training, use `msrun` and along with `--distributed`:

```shell
msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
    train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=1024 --center_crop --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --validation_prompt="a man in a green coat holding two swords" \
  --train_text_encoder \
  --distributed \
  --output_dir="sdxl-onepiece-model-lora-$(date +%Y%m%d%H%M%S)"
```

### Performance

For the above training example, we record the training speed as follows.

| Method | NPUs | Global <br/>Batch size | Resolution   | Precision | Graph Compile | Speed <br/>(s/step) | FPS <br/>(img/s) |
|--------|------|------------------------|--------------|-----------|---------------|---------------------|------------------|
| lora   | 1    | 1*1                    | 1024x1024    | FP16      | 15~20 mins    | 0.951               | 1.05             |
| lora   | 1    | 1*1                    | 1024x1024    | BF16      | 15~20 mins    | 0.994               | 1.01             |
| lora   | 1    | 1*1                    | 1024x1024    | FP32      | 15~20 mins    | 1.89                | 0.53             |

### Inference

If the LoRA weights you want to use is from huggingface, you can replace the following model_path like `model_path = "takuoko/sd-pokemon-model-lora-sdxl"`. Once you have trained a model using above command, the inference can be done simply using the `DiffusionPipeline` after loading the trained LoRA weights. You
need to pass the `output_dir` for loading the LoRA weights which, in this case, is `sdxl-onepiece-model-lora`.

```python
import mindspore as ms
from mindone.diffusers import DiffusionPipeline

model_path = "sdxl-onepiece-model-lora"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16)
pipe.load_lora_weights(model_path)

prompt = "a guy with green hair"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)[0][0]
image.save("onepiece.png")
```

We trained 8.5k steps based on the OnePiece dataset. Here are some of the results of the lora fine-tuning.

|                                                                       a cartoon character with a sword                                                                       |                                                                  a girl with a mask on her face                                                                   |                                                                  a guy with green hair                                                                   |                                                                  a lion sitting on the ground                                                                   |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_lora_infer/a_cartoon_character_with_a_sword.png?raw=true" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_lora_infer/a_girl_with_a_mask_on_her_face.png" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_lora_infer/a_guy_with_green_hair.png" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_lora_infer/a_lion_sitting_on_the_ground.png" width=224> |

|                                                                  a man holding a book                                                                   |                                                                  a man in a cowboy hat                                                                   |                                                                  a man in a hat and jacket                                                                   |                                                                  a man in a yellow coat                                                                   |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_lora_infer/a_man_holding_a_book.png" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_lora_infer/a_man_in_a_cowboy_hat.png" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_lora_infer/a_man_in_a_hat_and_jacket.png" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_lora_infer/a_man_in_a_yellow_coat.png" width=224> |

|                                                                  a man sitting in a chair                                                                   |                                                                  a man with a big beard                                                                   |                                                                  a man with green hair and a white shirt                                                                   |                                                                  a smiling woman in a helmet                                                                   |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_lora_infer/a_man_sitting_in_a_chair.png" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_lora_infer/a_man_with_a_big_beard.png" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_lora_infer/a_man_with_green_hair_and_a_white_shirt.png" width=224> | <img src="https://github.com/liuchuting/mindone/blob/image/examples/diffusers/text_to_image/images/sdxl_lora_infer/a_smiling_woman_in_a_helmet.png" width=224> |
