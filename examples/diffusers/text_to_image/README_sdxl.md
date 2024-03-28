# Stable Diffusion XL text-to-image fine-tuning

The `train_text_to_image_sdxl.py` script shows how to fine-tune Stable Diffusion XL (SDXL) on your own dataset.

🚨 This script is experimental. The script fine-tunes the whole model and often times the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyperparameters to get the best result on your dataset. 🚨

## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
https://github.com/mindspore-lab/mindone
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
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

python train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --max_train_steps=10000 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --validation_prompt="a cute Sundar Pichai creature" --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="sdxl-pokemon-model-$(date +%Y%m%d%H%M%S)"
```

**Notes**:

*  The `train_text_to_image_sdxl.py` script pre-computes text embeddings and the VAE encodings and keeps them in memory. While for smaller datasets like [`lambdalabs/pokemon-blip-captions`](https://hf.co/datasets/lambdalabs/pokemon-blip-captions), it might not be a problem, it can definitely lead to memory problems when the script is used on a larger dataset. For those purposes, you would want to serialize these pre-computed representations to disk separately and load them during the fine-tuning process. Refer to [this PR](https://github.com/huggingface/diffusers/pull/4505) for a more in-depth discussion.
* The training script is compute-intensive and only runs on an Ascend 910*.
* The training command shown above performs intermediate quality validation in between the training epochs. `--report_to`, `--validation_prompt`, and `--validation_epochs` are the relevant CLI arguments here.
* SDXL's VAE is known to suffer from numerical instability issues. This is why we also expose a CLI argument namely `--pretrained_vae_model_name_or_path` that lets you specify the location of a better VAE (such as [this one](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)).

### Inference

```python
from mindone.diffusers import DiffusionPipeline
import mindspore

model_path = "you-model-id-goes-here" # <-- change this
pipe = DiffusionPipeline.from_pretrained(model_path, mindspore_dtype=mindspore.float16)

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)[0][0]
image.save("pokemon.png")
```
