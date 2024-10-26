# DreamBooth finetune for Stable Diffusion XL (SDXL)

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method for personalizing text-to-image diffusion models, with just a few images (3~5) of a subject and its name as a Unique Identifier. During fine-tuning, a class-specific prior-preservation loss is applied in parallel, which leverages the semantic prior that the model has on the class and encourages output diversity.

For example, we have 5 images of a specific [dog](https://github.com/google/dreambooth/tree/main/dataset/dog) belonging to the prompt "a sks dog" for fine-tuning, where "sks" is a Unique Identifier. In parallel, images of general dogs, which are the class images in a text prompt "a dog", are inputted, so that the models will not forget other dogs' look.

The `train_dreambooth.py` script implements DreamBooth finetune for SDXL.

## Requirements

| mindspore      | ascend driver | firmware    | cann toolkit/kernel |
|:--------------:|:-------------:|:-----------:|:-------------------:|
| 2.2.10～2.2.12 | 23.0.3        | 7.1.0.5.220 | 7.0.0.beta1         |


## Pretrained models

Please follow SDXL [weight conversion](./preparation.md#convert-pretrained-checkpoint) for detailed steps and put the pre-trained weight to `./checkpoints/`.

The scripts automatically download the clip tokenizer. If you have network issues with it, [FAQ Qestion 5](./faq_cn.md#5-连接不上huggingface-报错-cant-load-tokenizer-for-openaiclip-vit-large-patch14) helps.

## Datasets preparation

The finetuning dataset should contain 3-5 images from the same subject in the same folder.

```text
dir
├── img1.jpg
├── img2.jpg
├── img3.jpg
├── img4.jpg
└── img5.jpg
```

You can find images of different classes in [Google/DreamBooth](https://github.com/google/dreambooth/tree/main). Here we use two examples, [dog](https://github.com/google/dreambooth/tree/main/dataset/dog) and [dog6](https://github.com/google/dreambooth/tree/main/dataset/dog6). They are shown as,

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/961bdff6-f565-4cf2-85ce-e59c6ed547f3" width=800 />
</p>
<p align="center">
  <em> Figure 1. dog example: the five images from the subject dog for finetuning. </em>
</p>

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/a5bef2fc-b613-46de-8021-3e489dd663a1" width=800 />
</p>
<p align="center">
  <em> Figure 2. dog6 example: the five images from the subject dog for finetuning. </em>
</p>


## Finetuning

Before running the fintune scripts `train_dreambooth.py`, please specify the arguments that might vary from users.

* `--instance_data_path=/path/to/finetuning_data `
* `--class_data_path=/path/to/class_image `
* `--weight=/path/to/pretrained_model`
* `--save_path=/path/to/save_models`

Modify other arguments in the shell when running the command or the hyper-parameters in the config file `sd_xl_base_finetune_dreambooth_lora_910*.yaml` if needed.

Run with multiple NPUs (for example, 4) training,

```shell
mpirun --allow-run-as-root -n 4 python train_dreambooth.py \
  --config configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --instance_data_path /path/to/finetuning_data \
  --instance_prompt "A photo of a sks dog" \
  --class_data_path /path/to/class_image \
  --class_prompt "A photo of a dog" \
  --ms_mode 0 \
  --save_ckpt_interval 500 \
  --is_parallel True \
  --device_target Ascend
```

Launch a standalone training,

```shell
python train_dreambooth.py \
  --config configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --instance_data_path /path/to/finetuning_data \
  --instance_prompt "A photo of a sks dog" \
  --class_data_path /path/to/class_image \
  --class_prompt "A photo of a dog" \
  --gradient_accumulation_steps 4 \
  --ms_mode 0 \
  --save_ckpt_interval 500 \
  --device_target Ascend
```

Our implementation is trained with prior-preservation loss, which avoids overfitting and language drift. We first generate images using the pertained model with a class prompt, and input those data in parallel with our data during finetuning. The `num_class_images` in the arguments of `train_dreambooth.py`  specifies the number of class images for prior-preservation. If not enough images are present in `class_image_path`, additional images will be sampled with `class_prompt`. And you would need to relaunch the training using the command above when sampling is finished. It takes about 25 minutes to sample 50 class images.

**Training with the two text encoders in SDXL is supported** by replacing the config of training commands above with `configs/training/sd_xl_base_finetune_dreambooth_textencoder_lora_910b.yaml`. In our experiment, training without text encoders yields better generation results.


## Inference

Training above get finetuned weights in the specified `save_path`. Assume that the pretrained ckpt path is `checkpoints/sd_xl_base_1.0_ms.ckpt` and the trained lora ckpt path is `runs/SDXL_base_1.0_1000_lora.ckpt`. The two paths are separated by a comma without space and passed to the inference command.

```shell
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt,runs/SDXL_base_1.0_1000_lora.ckpt \
  --prompt "a sks dog swimming in a pool" \
  --device_target Ascend
```

Examples of generated images with the DreamBooth model using different prompts are shown below.

The [dog](https://github.com/google/dreambooth/tree/main/dataset/dog) example finetuning results,

* "A photo of a sks dog swimming in a pool"

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/0ddf4ce1-4177-44c0-84bd-2b15c0e2f6f4" width=700 />



The [dog6](https://github.com/google/dreambooth/tree/main/dataset/dog6) example finetuning results,

* "A photo of a sks dog in a bucket"

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/5144b904-329c-4d83-aa4b-c2f4ecd60ea0" width=700 />



* "A photo of a sks dog in a doghouse"

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/6b2a6656-10a0-4d9d-8542-a9fa0527bc8a" width=700 />


## Performance

Experiments are tested on ascend 910* with mindspore 2.2.12 graph mode. Experiments use the Dreambooth method with LoRA and enable UNet training only.

| model name    | card | bs * grad accu. |   resolution       |  flash attention |  ms/step  | fps |
|:---------------:|:----------------:|:----------------:|------------------| :--: |:----------------:|:--: |
| SDXL-Base     |      1            |      1x1             |     1024x1024         | ON |       1280       |0.78 |
> fps: images per second during training. average training time (s/step) = batch_size / fps
