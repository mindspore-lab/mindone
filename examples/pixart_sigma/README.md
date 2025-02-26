# PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation (Mindspore)

This repo contains Mindspore model definitions, pre-trained weights and inference/sampling code for the [paper](https://arxiv.org/abs/2403.04692) exploring Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation. You can find more visualizations on the [official project page](https://pixart-alpha.github.io/PixArt-sigma-project/).

## Contents

- Main
    - [Training](#vanilla-finetune)
    - [Inference](#getting-start)
    - [Launch Demo](#running-the-demo-with-gradio)
    - [Use diffusers](#integration-in-diffusers)
- Benchmark
    - [Training](#training)
    - [Inference](#inference)
- Guidance
    - [LoRA Finetuning](#lora-finetuning)

## What's New
- 2024-09-05
    - Support fine-tuning and inference for Pixart-Sigma models.

## Requirements

| mindspore | ascend driver | firmware    | cann toolkit/kernel |
|:-----------:|:---------------:|:-------------:|:---------------------:|
| 2.3.1     | 24.1.RC2      | 7.3.0.1.231 | 8.0.RC2.beta1       |


Python: 3.9 or later.

Then run `pip install -r requirements.txt` to install the necessary packages.


## Getting Start

### Downloading Pretrained Checkpoints

We refer to the [official repository of PixArt-sigma](https://github.com/PixArt-alpha/PixArt-sigma) for pretrained checkpoints downloading.

After downloading the `PixArt-Sigma-XL-2-256x256.pth` and `PixArt-Sigma-XL-2-{}-MS.pth`, please place it under the `models/` directory, and then run `tools/convert.py` for each checkpoint separately. For example, to convert `models/PixArt-Sigma-XL-2-1024-MS.pth`, you can run:

```bash
python tools/convert.py --source models/PixArt-Sigma-XL-2-1024-MS.pth --target models/PixArt-Sigma-XL-2-1024-MS.ckpt
```

> Note: You must have an environment with `PyTorch` installed to run the conversion script.

In addition, please download the [VAE checkpoint](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae), [T5 checkpoint](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/text_encoder), [T5 Tokenizer](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/tokenizer) and put them under `models` directory.


After conversion, the checkpoints under `models/` should be like:
```bash
models/
├── PixArt-Sigma-XL-2-256x256.ckpt
├── PixArt-Sigma-XL-2-512-MS.ckpt
├── PixArt-Sigma-XL-2-1024-MS.ckpt
├── PixArt-Sigma-XL-2-2K-MS.ckpt
├── vae/
├── tokenizer/
└── text_encoder/
```

### Sampling using Pretrained model

You can then run the sampling using `sample.py`. For examples, to sample a 512x512 resolution image, you may run

```bash
python sample.py -c configs/inference/pixart-sigma-512-MS.yaml --prompt "your magic prompt"
```

For higher resolution images, you can choose either `configs/inference/pixart-sigma-1024-MS.yaml` or `configs/inference/pixart-sigma-2K-MS.yaml`.

And to sample an image with a varying aspect ratio, you need to use the flag `--image_width` and `--image_width`. For example, to sample a 512x1024 image, you may run

```bash
python sample.py -c configs/inference/pixart-sigma-1024-MS.yaml --prompt "your magic prompt" --image_width 1024 --image_height 512
```

The following demo image is generated using the following command:

```bash
python sample.py -c configs/inference/pixart-sigma-1024-MS.yaml --image_width 1024 --image_height 512 --seed 1024 --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
```
<p align="center"><img width="1024" src="https://github.com/user-attachments/assets/bcf12b8d-1077-451b-a6ae-51bbf3c8de7a"/></p>

You can also generate batch images using a text file with, where the file stores prompts separated by `\n`. Use the following command to generate images:

```bash
python sample.py -c configs/inference/pixart-sigma-1024-MS.yaml --prompt_path path_to_yout_text_file --batch_size 4
```

For more detailed usage of the inference script, please run `python sample.py -h`.

### Vanilla Finetune

We support finetune PixArt-Σ model on 910* Ascend device.

#### Prepare the Dataset

- As an example, please download the `diffusiondb-pixelart` dataset from [this link](https://huggingface.co/datasets/jainr3/diffusiondb-pixelart). The dataset is a subset of the larger DiffusionDB 2M dataset, which has been transformed into pixel-style art.

- Once you have the dataset, create a label JSON file in the following format:
```json
[
    {
        "path": "file1.png",
        "prompt": "a beautiful photorealistic painting of cemetery urbex unfinished building building industrial architecture...",
        "sharegpt4v": "*caption from ShareGPT4V*",
        "height": 512,
        "width": 512,
        "ratio": 1.0,
    },
]
```
- Remember to
    - Replace `file1.png` with the actual image file path.
    - The `prompt` field contains a description of the image.
    - If you have captions generated from ShareGPT4V, add them to the `sharegpt4v` field. Otherwise, copy the label from the `prompt` line.
    - `height` and `width` field corresponds to the image height and width, and `ratio` corresponds to the value of `height` / `width`.

#### Finetune the Model:

Use the following command to start the finetuning process:

```bash
python train.py \
    -c configs/train/pixart-sigma-512-MS.yaml \
    --json_path path_to_your_label_file \
    --image_dir path_to_your_image_directory
```
- Remember to
    - Replace `path_to_your_label_file` with the actual path to your label JSON file.
    - Replace `path_to_your_image_directory` with the directory containing your images.

For more detailed usage of the training script, please run `python train.py -h`.

Once you have finishsed your training, you can run sampling with your own checkpoint file with the following command

```bash
python sample.py -c configs/inference/pixart-sigma-512-MS.yaml --prompt "your magic prompt" --checkpoint path_to_your_checkpoint_file
```

#### Distributed Training (Optional):

You can launch distributed training using multiple Ascend 910* Devices:

```bash
msrun --worker_num=8 --local_worker_num=8 --log_dir="log" train.py \
    -c configs/train/pixart-sigma-512-MS.yaml \
    --json_path path_to_your_label_file \
    --image_dir path_to_your_image_directory \
    --use_parallel True
```
- Remember to
    - Replace `path_to_your_label_file` with the actual path to your label JSON file.
    - Replace `path_to_your_image_directory` with the directory containing your images.

#### Finetune Result

We use the first 1,600 images for training and the remaining 400 images for testing. The experiment is conducted on two 910* NPUs based on the [configuration](configs/train/pixart-sigma-512-MS.yaml). We evaluate the model’s performance using the [FID score](https://github.com/mindspore-lab/mindone/tree/master/examples/stable_diffusion_v2/tools/eval).

Below is the FID score curve
<p align="center"><img width="512" src="https://github.com/user-attachments/assets/b3d74961-15f7-4836-9b26-db3b470c3565"/></p>

Followed by some generated images using the testing prompts.
<p align="center"><img width="1024" src="https://github.com/user-attachments/assets/b9ba152d-bbf0-46c2-af10-ba8066b92486"/></p>

### Running the Demo with Gradio
First install gradio by
``` bash
pip install gradio
```
**Basic Usage**: Run the demo with the recommended configuration using the following command:

```bash
python gradio_demo.py \
    --image_height 1024 \
    --image_width 1024 \
    --sample_size 128 \
    --checkpoint "models/PixArt-Sigma-XL-2-1024-MS.ckpt" \
    --vae_root "models/vae" \
    --text_encoder_root "models/text_encoder" \
    --tokenizer_root "models/tokenizer" \
    --sd_scale_factor 0.13025 \
    --enable_flash_attention True \
    --dtype "fp16"
```

**Parameter Explanation**
- `--image_height`: Output image height (1024 recommended)
- `--image_width`: Output image width (1024 recommended)
- `--sample_size`: Size of latent samples (128 for 1024px images)
- `--checkpoint`: Path to the PixArt model checkpoint
- `--vae_root`: Directory containing VAE model files
- `--text_encoder_root`: Directory containing text encoder model
- `--tokenizer_root`: Directory containing tokenizer files
- `--sd_scale_factor`: VAE scaling factor (0.13025 recommended)
- `--enable_flash_attention`: Enable flash attention for better performance
- `--dtype`: Model precision ("fp16" recommended for GPU)

**Using the Web Interface**

After running the command, the demo will launch a web interface accessible at:
```
http://localhost:7788
```

The interface allows you to:
1. Enter text prompts
2. Choose sampling method (dpm, ddim, iddpm)
3. Adjust sampling steps
4. Modify guidance scale
5. Set random seed
6. Generate images
<p align="center"><img width="1024" src="https://github.com/itruonghai/mindone-asset/blob/main/gradio_demo.png?raw=true"/></p>

### Integration in diffusers
```python
from mindone.diffusers import DiffusionPipeline, Transformer2DModel, PixArtSigmaPipeline
import mindspore as ms

transformer = Transformer2DModel.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    subfolder='transformer',
    mindspore_dtype=ms.float16,
    use_safetensors=True,
)
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    transformer=transformer,
    mindspore_dtype=ms.float16,
    use_safetensors=True,
)
prompt =["Whimsical forest fairy resting on a mossy toadstool, surrounded by glowing fireflies.",
        "Brass and gear-laden mechanical owl soaring gracefully through a cloudy, steampunk-inspired cityscape.",
        "Abandoned, weathered spaceship drifting silently through a field of sparkling asteroids.",
        "Warm light spilling out from the windows of a cozy cottage nestled in a snowy, pine-filled forest.",
        "Majestic, dragon-like creature gliding over a rugged, fantastical mountain range, casting a dramatic shadow below."]
image = pipe(prompt)[0]
for i in range(5):
    image[i].save("./prompt{}.png".format(i))
```
Generated image from the code.
<p align="center"><img width="1024" src="https://github.com/itruonghai/mindone-asset/blob/main/pixart-sigma.png?raw=true"/></p>

## Performance

### Training Performance

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode

| model name   | cards | batch size | resolution   | recompute | sink | jit level |graph compile | s/step | img/s | config                                                               |
|:------------:|:-----:|:----------:|:------------:|:----------:|:---------:|:---------:|:---------:|:---------:|:-------------:|:--------------------------------------------------------------------|
| PixArt-Sigma | 4     | 64         | 256x256      | ON        | OFF       | O1        |3~5 mins      |  2.907s    | 88.1          | [pixart-sigma-256x256.yaml](configs/train/pixart-sigma-256x256.yaml) |
| PixArt-Sigma | 4     | 32         | 512 (multi)  | ON        | OFF       | O1 |3~5 mins       | 3.817s    | 33.5          | [pixart-sigma-512-MS.yaml](configs/train/pixart-sigma-512-MS.yaml)   |
| PixArt-Sigma | 4     | 12         | 1024 (multi) | ON        | OFF       | O1  |3~5 mins      | 7.042s    | 6.8           | [pixart-sigma-1024-MS.yaml](configs/train/pixart-sigma-1024-MS.yaml) |
| PixArt-Sigma | 4     | 1          | 2048 (multi) | ON        | OFF       | O1  |3~5 mins      | 8.772s    | 0.5           | [pixart-sigma-2K-MS.yaml](configs/train/pixart-sigma-2K-MS.yaml)     |

> s/step: training time measured in the number of seconds for each training step.\
> imgs/s: images per second during training. imgs/s = cards * batch_size / step time

### Inference Performance

| model name   | cards |  batch size | resolution   | jit level |  graph compile | s/step |  recipe                                                                  |
|:------------:|:-----:|:-----------:|:------------:|:----------:|:---------:|:---------:|:------------------------------------------------------------------------|
| PixArt-Sigma | 1     | 1           | 256 x 256    | O1        | < 3 mins      | 0.055    | [yaml](configs/inference/pixart-sigma-256x256.yaml) |
| PixArt-Sigma | 1     | 1           | 512 x 512    | O1        | < 3 mins      | 0.063    | [yaml](configs/inference/pixart-sigma-512-MS.yaml)   |
| PixArt-Sigma | 1     | 1           | 1024 x 1024  | O1        | < 3 mins      | 0.202    | [yaml](configs/inference/pixart-sigma-1024-MS.yaml) |
| PixArt-Sigma | 1     | 1           | 2048 x 2048  | O1        | < 3 mins      | 1.754    | [yaml](configs/inference/pixart-sigma-2K-MS.yaml)     |


## LoRA Finetuning
We provide the train and inference of Pixart-Alpha/Pixart-Sigma in `pixart/lora-finetuning`

### LoRA Training
You can fine-tune PixArt-alpha or PixArt-Sigma models using LoRA. The base models are available on [Hugging Face](https://huggingface.co/PixArt-alpha).

Below is an example script for training on the Pokemon dataset using PixArt-XL-2-512x512. Note that for PixArt-Sigma, increase the `max-token-length` to 300.


```bash
dataset_id=svjack/pokemon-blip-captions-en-zh
caption_column=en_text
model_id=PixArt-alpha/PixArt-XL-2-512x512

python  pixart/lora-finetuning/train_pixart_lora.py \
    --pretrained_model_name_or_path=$model_id \
    --dataset_name=$dataset_id \
    --caption_column=$caption_column \
    --resolution=512   --random_flip \
    --train_batch_size=4 \
    --num_train_epochs=80   --checkpointing_steps=200 \
    --learning_rate=1e-04   --lr_scheduler="constant" \
    --lr_warmup_steps=0   --seed=42 \
    --mixed_precision="fp16" \
    --output_dir="output/pixart-pokemon-model" \
    --validation_prompt="cute dragon creature" \
    --report_to="tensorboard" \
    --checkpoints_total_limit=10 \
    --validation_epochs=5 \
    --max_token_length=120
```

### Inference
After training, you can generate images using your fine-tuned model. Here's how to use it:
```bash
python pixart/lora-finetuning/inference_pixart_lora.py \
    --model_id="PixArt-alpha/PixArt-XL-2-512x512" \
    --lora_path="output/pixart-pokemon-model/lora_checkpoint_path" \
    --prompt="A grass-type Pokemon in a forest, highly detailed" \
    --output_path="./generated_pokemon.png"
```
The script accepts the following parameters:
- `--model_id`: Base model ID from Hugging Face or local path (default: "PixArt-alpha/PixArt-XL-2-512x512")
- `--lora_path`: Path to your trained LoRA weights directory (required)
- `--prompt`: Text description for the image you want to generate
- `--output_path`: Where to save the generated image (default: "./pokemon.png")


Here is some examples
<p align="center"><img width="1024" src="https://github.com/itruonghai/mindone-asset/blob/main/pixart_lora.png?raw=true"/></p>

#### LoRA Finetuning Performance



| model name   | cards |  batch size | resolution   | jit level |  graph compile | s/step |  img/s                                                                  |
|:------------:|:-----:|:-----------:|:------------:|:----------:|:---------:|:---------:|:------------------------------------------------------------------------|
| PixArt-Alpha | 1     | 4           | 512 x 512    | O1        | < 3 mins      | 0.055    | 72.72 |
| PixArt-Sigma | 1     | 4           | 512 x 512    | O1        | < 3 mins      | 0.065    | 61.54 |


# References

[1] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, Zhenguo Li. PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation. arXiv:2403.04692, 2024.
