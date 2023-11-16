# LoRA for Stable Diffusion Finetuning
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



## Get Started

**MindONE** supports LoRA finetuning for Stable Diffusion models based on MindSpore and Ascend platforms.

### Preparation

#### Dependency

Please make sure the following frameworks are installed.

- mindspore >= 1.9  [[install](https://www.mindspore.cn/install)] (2.0 is recommended for the best performance.)
- python >= 3.7
- openmpi 4.0.3 (for distributed training/evaluation)  [[install](https://www.open-mpi.org/software/ompi/v4.0/)]

Install the dependent packages by running:
```shell
pip install -r requirements.txt
```

#### Pretrained Models

Please download the pretrained [SD2.0-base checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt) and put it under `models/` folder.


#### Text-image Dataset Preparation

The text-image pair dataset for finetuning should follow the file structure below

```text
dir
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── img_txt.csv
```

img_txt.csv is the annotation file in the following format
```text
dir,text
img1.jpg,a cartoon character with a potted plant on his head
img2.jpg,a drawing of a green pokemon with red eyes
img3.jpg,a red and white ball with an angry look on its face
```

For convenience, we have prepared two public text-image datasets obeying the above format.

- [pokemon-blip-caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 833 pokemon-style images with BLIP-generated captions.
- [Chinese-art blip caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 100 chinese art-style images with BLIP-generated captions.

To use them, please download `pokemon_blip.zip` and `chinese_art_blip.zip` from the [openi dataset website](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets). Then unzip them on your local directory, e.g. `./datasets/pokemon_blip`.

### LoRA Finetune

We will use the `train_text_to_image.py` script and set argument `use_lora=True` for LoRA finetuning.
Before running, please modify the following arguments to your local path in the shell or in the config file `train_config_lora_v2.yaml`:

* `--data_path=/path/to/data`
* `--output_path=/path/to/save/output_data`
* `--pretrained_model_path=/path/to/pretrained_model`

Then, execute the script to launch finetuning:

```shell
python train_text_to_image.py \
    --train_config "configs/train/train_config_lora_v2.yaml" \
    --data_path "datasets/pokemon_blip/train" \
    --output_path "output/lora_pokemon/txt2img" \
    --pretrained_model_path "models/sd_v2_base-57526ee4.ckpt"
```

> Note: to modify other important hyper-parameters, please refer to training config file `train_config_lora_v2.yaml`.

After training, the lora checkpoint will be saved in `{output_path}/ckpt/txt2img/ckpt/rank_0/sd-72.ckpt` by default, which only contains the LoRA parameters and is small.

Below are some arguments that you may want to tune for a better performance on your dataset:


- `lora_rank`: the rank of the low-rank matrices in lora params.
- `train_batch_size`: the number of batch size for training.
- `start_learning_rate` and `end_learning_rate`: the initial and end learning rates for training.
- `epochs`: the number of epochs for training.
- `use_ema`: whether use EMA for model smoothing
> Note that the default learning rate for LoRA is 1e-4, whichis larger that vanilla finetuning (~1e-5).

For more argument illustration, please run `python train_text_to_image.py -h`.


#### Config for v-prediction (Experimental)

By default, the target of LDM model is to predict the noise of the diffusion process (called `eps-prediction`). `v-prediction` is another prediction type where the `v-parameterization` is involved (see section 2.4 in [this paper](https://imagen.research.google/video/paper.pdf)) and is claimed to have better convergence and numerical stability.

To switch from `eps-prediction` to `v-prediction`, please modify `configs/v2-train.yaml` as follows.

```yaml
#parameterization: "eps"
parameterization: "velocity"
```

### Inference

To perform text-to-image generation with the finetuned lora checkpoint, please run

```shell
python text_to_image.py \
        --prompt "A drawing of a fox with a red tail" \
        --use_lora True \
        --lora_ckpt_path {path/to/lora_checkpoint_after_finetune}
```

Please update `lora_ckpt_path` according to your finetune settings.

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

### Evaluation

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

Here are the evaluation results for our implementation.

<div align="center">

| Pretrained Model          | Context | Dataset          | Finetune Method | Sampling Algo.                   | FID (Ours) &#8595; | FID (Diffuser) &#8595; | CLIP Score (Ours) &#8593; | CLIP Score (Diffuser) &#8593; |
|---------------------------|----------------|----------|-----------------|----------------------------------|--------------------|------------------------|---------------------------|-------------------------------|
| stable_diffusion_2.0_base | 910Ax1-MS2.0-G | pokemon_blip     | LoRA            | DPM Solver (scale: 9, steps: 15) | 108                | 106                    | 30.8                      | 31.6                          |
| stable_diffusion_2.0_base | 910Ax1-MS2.0-G | chinese_art_blip | LoRA            | DPM Solver (scale: 4, steps: 15) | 257                | 254                    | 33.6                      | 33.2                          |

</div>

> Note that these numbers can not reflect the generation quality comprehensively!! A visual evaluation is also necessary.
> Context: Training context denoted as {device}x{pieces}-{MS version}{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910A NPU using graph mode.

## Notes

- Apply LoRA to Text-encoder

By default, LoRA fintuning is only applied to UNet. To finetune the text encoder with LoRA as well, please pass `--lora_ft_text_encoder=True` to the finetuning script (`train_text_to_image.py`) and inference script (`text_to_image.py`).


## Reference
[1] [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
