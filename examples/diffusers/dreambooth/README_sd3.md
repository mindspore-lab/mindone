# DreamBooth training example for Stable Diffusion 3 (SD3)

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few (3~5) images of a subject.

The `train_dreambooth_sd3.py` script shows how to implement the training procedure and adapt it for [Stable Diffusion 3](https://huggingface.co/papers/2403.03206). We also provide a LoRA implementation in the `train_dreambooth_lora_sd3.py` script.

> [!NOTE]  
> As the model is gated, before using it with diffusers you first need to go to the [Stable Diffusion 3 Medium Hugging Face page](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in:

```bash
huggingface-cli login
```

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

<details>
  <summary>
  WIP: run example in LoRA+DreamBooth instead
  </summary>

  Now, we can launch training using:

  ```bash
  export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
  export INSTANCE_DIR="dog"
  export OUTPUT_DIR="trained-sd3"

  python train_dreambooth_sd3.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
    --instance_prompt="a photo of sks dog" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=500 \
    --validation_prompt="A photo of sks dog in a bucket" \
    --validation_epochs=25 \
    --seed="0"
  ```

  To better track our training experiments, we're using the following flags in the command above:

  * `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

</details>


## LoRA + DreamBooth

[LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is a popular parameter-efficient fine-tuning technique that allows you to achieve full-finetuning like performance but with a fraction of learnable parameters.

To perform DreamBooth with LoRA, run:

```bash
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-sd3-lora"

python train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
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
