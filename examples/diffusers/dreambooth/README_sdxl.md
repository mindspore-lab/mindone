# DreamBooth training example for Stable Diffusion XL (SDXL)

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few (3~5) images of a subject.

The `train_dreambooth_lora_sdxl.py` script shows how to implement the training procedure and adapt it for [Stable Diffusion XL](https://huggingface.co/papers/2307.01952).

> 💡 **Note**: For now, we only allow DreamBooth fine-tuning of the SDXL UNet via LoRA. LoRA is a parameter-efficient fine-tuning technique introduced in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*.


## Running locally with MindSpore

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/mindspore-lab/mindone
cd mindone
pip install -e ".[training]"
```


### Dog toy example

Now let's get our dataset. For this example we will use some dog images: https://huggingface.co/datasets/diffusers/dog-example.

Let's first download it locally:

```python
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

This will also allow us to push the trained LoRA parameters to the Hugging Face Hub platform.

Now, we can launch training using:

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="lora-trained-xl"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

python train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0"
```

To better track our training experiments, we're using the following flags in the command above:

* `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

### Inference

Once training is done, we can perform inference like so:

```python
from huggingface_hub.repocard import RepoCard
from mindone.diffusers import DiffusionPipeline
import mindspore as ms

lora_model_id = "<lora-sdxl-dreambooth-id>"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, mindspore_dtype=ms.float16)
pipe.load_lora_weights(lora_model_id)
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25)[0][0]
image.save("sks_dog.png")
```

We can further refine the outputs with the [Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0):

```python
from huggingface_hub.repocard import RepoCard
from mindone.diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import mindspore as ms
import numpy as np

lora_model_id = "<lora-sdxl-dreambooth-id>"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

# Load the base pipeline and load the LoRA parameters into it.
pipe = DiffusionPipeline.from_pretrained(base_model_id, mindspore_dtype=ms.float16)
pipe.load_lora_weights(lora_model_id)

# Load the refiner.
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", mindspore_dtype=ms.float16, use_safetensors=True, variant="fp16"
)

prompt = "A picture of a sks dog in a bucket"
generator = np.random.Generator(np.random.PCG64(seed=0))

# Run inference.
image = pipe(prompt=prompt, output_type="latent", generator=generator)[0][0]
image = refiner(prompt=prompt, image=image[None, :], generator=generator)[0][0]
image.save("refined_sks_dog.png")
```

Here's a side-by-side comparison of the with and without Refiner pipeline outputs:

| Without Refiner | With Refiner |
|---|---|
| ![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/sks_dog.png) | ![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/refined_sks_dog.png) |

### Training with text encoder(s)

Alongside the UNet, LoRA fine-tuning of the text encoders is also supported. To do so, just specify `--train_text_encoder` while launching training. Please keep the following points in mind:

* SDXL has two text encoders. So, we fine-tune both using LoRA.
* When not fine-tuning the text encoders, we ALWAYS precompute the text embeddings to save memory.

### Specifying a better VAE

SDXL's VAE is known to suffer from numerical instability issues. This is why we also expose a CLI argument namely `--pretrained_vae_model_name_or_path` that lets you specify the location of a better VAE (such as [this one](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)).


## Notes

In our experiments, we found that SDXL yields good initial results without extensive hyperparameter tuning. For example, without fine-tuning the text encoders and without using prior-preservation, we observed decent results. We didn't explore further hyper-parameter tuning experiments, but we do encourage the community to explore this avenue further and share their results with us 🤗


## Results

You can explore the results from a couple of our internal experiments by checking out this link: [https://wandb.ai/sayakpaul/dreambooth-lora-sd-xl](https://wandb.ai/sayakpaul/dreambooth-lora-sd-xl). Specifically, we used the same script with the exact same hyperparameters on the following datasets:

* [Dogs](https://huggingface.co/datasets/diffusers/dog-example)
* [Starbucks logo](https://huggingface.co/datasets/diffusers/starbucks-example)
* [Mr. Potato Head](https://huggingface.co/datasets/diffusers/potato-head-example)
* [Keramer face](https://huggingface.co/datasets/diffusers/keramer-face-example)


## Conducting EDM-style training

It's now possible to perform EDM-style training as proposed in [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364).

For the SDXL model, simple set:

```diff
+  --do_edm_style_training \
```

Other SDXL-like models that use the EDM formulation, such as [playgroundai/playground-v2.5-1024px-aesthetic](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic), can also be DreamBooth'd with the script. Below is an example command:

```bash
python train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="playgroundai/playground-v2.5-1024px-aesthetic"  \
  --instance_data_dir="dog" \
  --output_dir="dog-playground-lora" \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0"
```

> [!CAUTION]
> Min-SNR gamma is not supported with the EDM-style training yet. When training with the PlaygroundAI model, it's recommended to not pass any "variant".
