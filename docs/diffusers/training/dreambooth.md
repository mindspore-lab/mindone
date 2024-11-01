<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DreamBooth

[DreamBooth](https://huggingface.co/papers/2208.12242) is a training technique that updates the entire diffusion model by training on just a few images of a subject or style. It works by associating a special word in the prompt with the example images.

If you want to reduce memory footprint, you should try enabling the `gradient_checkpointing` and `mixed_precision` parameters in the training command. You can also reduce your memory footprint by using memory-efficient attention with [xFormers](../optimization/xformers.md).

This guide will explore the [train_dreambooth.py](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth.py) script to help you become more familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install .
```

Lastly, if you want to train a model on your own dataset, take a look at the [Create a dataset for training](create_dataset.md) guide to learn how to create a dataset that works with the training script.

!!! tip

    The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth.py) and let us know if you have any questions or concerns.

## Script parameters

!!! warning

    DreamBooth is very sensitive to training hyperparameters, and it is easy to overfit. Read the [Training Stable Diffusion with Dreambooth using 🧨 Diffusers](https://huggingface.co/blog/dreambooth) blog post for recommended settings for different subjects to help you choose the appropriate hyperparameters.

The training script offers many parameters for customizing your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth.py#L128) function. The parameters are set with default values that should work pretty well out-of-the-box, but you can also set your own values in the training command if you'd like.

For example, to train in the bf16 format:

```bash
python train_dreambooth.py \
    --mixed_precision="bf16"
```

Some basic and important parameters to know and specify are:

- `--pretrained_model_name_or_path`: the name of the model on the Hub or a local path to the pretrained model
- `--instance_data_dir`: path to a folder containing the training dataset (example images)
- `--instance_prompt`: the text prompt that contains the special word for the example images
- `--train_text_encoder`: whether to also train the text encoder
- `--output_dir`: where to save the trained model
- `--push_to_hub`: whether to push the trained model to the Hub
- `--checkpointing_steps`: frequency of saving a checkpoint as the model trains; this is useful if for some reason training is interrupted, you can continue training from that checkpoint by adding `--resume_from_checkpoint` to your training command

### Prior preservation loss

Prior preservation loss is a method that uses a model's own generated samples to help it learn how to generate more diverse images. Because these generated sample images belong to the same class as the images you provided, they help the model retain what it has learned about the class and how it can use what it already knows about the class to make new compositions.

- `--with_prior_preservation`: whether to use prior preservation loss
- `--prior_loss_weight`: controls the influence of the prior preservation loss on the model
- `--class_data_dir`: path to a folder containing the generated class sample images
- `--class_prompt`: the text prompt describing the class of the generated sample images

```bash
python train_dreambooth.py \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --class_data_dir="path/to/class/images" \
  --class_prompt="text prompt describing class"
```

### Train text encoder

To improve the quality of the generated outputs, you can also train the text encoder in addition to the UNet. This requires additional memory. If you have the necessary hardware, then training the text encoder produces better results, especially when generating images of faces. Enable this option by:

```bash
python train_dreambooth.py \
  --train_text_encoder
```

## Training script

DreamBooth comes with its own dataset classes:

- [`DreamBoothDataset`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth.py#L521): preprocesses the images and class images, and tokenizes the prompts for training
- [`PromptDataset`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth.py#L653): generates the prompt embeddings to generate the class images

If you enabled [prior preservation loss](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth.py#L734), the class images are generated here:

```py
sample_dataset = PromptDataset(args.class_prompt, num_new_images)
sample_dataloader = GeneratorDataset(
    sample_dataset, column_names=["example"], shard_id=args.rank, num_shards=args.world_size
).batch(batch_size=args.sample_batch_size)

for (example,) in tqdm(
    sample_dataloader_iter,
    desc="Generating class images",
    total=len(sample_dataloader),
    disable=not is_master(args),
):
    images = pipeline(example["prompt"].tolist())[0]
```

Next is the [`main()`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth.py#L715) function which handles setting up the dataset for training and the training loop itself. The script loads the [tokenizer](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth.py#L794), [scheduler and models](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth.py#L808):

```py
# Load the tokenizer
if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
elif args.pretrained_model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = text_encoder_cls.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
)

if model_has_vae(args):
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
else:
    vae = None

unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
)
```

Then, it's time to [create the training dataset](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth.py#L892) and DataLoader from `DreamBoothDataset`:

```py
train_dataset = DreamBoothDataset(
    instance_data_root=args.instance_data_dir,
    instance_prompt=args.instance_prompt,
    class_data_root=args.class_data_dir if args.with_prior_preservation else None,
    class_prompt=args.class_prompt,
    class_num=args.num_class_images,
    tokenizer=tokenizer,
    size=args.resolution,
    center_crop=args.center_crop,
    encoder_hidden_states=pre_computed_encoder_hidden_states,
    class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
    tokenizer_max_length=args.tokenizer_max_length,
)

train_dataloader = GeneratorDataset(
    train_dataset,
    column_names=["example"],
    shuffle=True,
    shard_id=args.rank,
    num_shards=args.world_size,
    num_parallel_workers=args.dataloader_num_workers,
).batch(
    batch_size=args.train_batch_size,
    per_batch_map=lambda examples, batch_info: collate_fn(examples, args.with_prior_preservation),
    input_columns=["example"],
    output_columns=["c1", "c2"] if args.pre_compute_text_embeddings else ["c1", "c2", "c3"],
    num_parallel_workers=args.dataloader_num_workers,
)
```

Lastly, the [training loop](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth.py#L1028) takes care of the remaining steps such as converting images to latent space, adding noise to the input, predicting the noise residual, and calculating the loss.

If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline.md) tutorial which breaks down the basic pattern of the denoising process.

## Launch the script

You're now ready to launch the training script! 🚀

For this guide, you'll download some images of a [dog](https://huggingface.co/datasets/diffusers/dog-example) and store them in a directory. But remember, you can create and use your own dataset if you want (see the [Create a dataset for training](create_dataset.md) guide).

```py
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

Set the environment variable `MODEL_NAME` to a model id on the Hub or a path to a local model, `INSTANCE_DIR` to the path where you just downloaded the dog images to, and `OUTPUT_DIR` to where you want to save the model. You'll use `sks` as the special word to tie the training to.

If you're interested in following along with the training process, you can periodically save generated images as training progresses. Add the following parameters to the training command:

```bash
--validation_prompt="a photo of a sks dog"
--num_validation_images=4
--validation_steps=100
```

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path_to_saved_model"

python train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub
```

Once training is complete, you can use your newly trained model for inference!

```py
from mindone.diffusers import DiffusionPipeline
import mindspore as ms

pipeline = DiffusionPipeline.from_pretrained("path_to_saved_model", mindspore_dtype=ms.float16, use_safetensors=True)
image = pipeline("A photo of sks dog in a bucket", num_inference_steps=50, guidance_scale=7.5)[0][0]
image.save("dog-bucket.png")
```

## LoRA

LoRA is a training technique for significantly reducing the number of trainable parameters. As a result, training is faster and it is easier to store the resulting weights because they are a lot smaller (~100MBs). Use the [train_dreambooth_lora.py](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth_lora.py) script to train with LoRA.

The LoRA training script is discussed in more detail in the [LoRA training](lora.md) guide.

## Stable Diffusion XL

Stable Diffusion XL (SDXL) is a powerful text-to-image model that generates high-resolution images, and it adds a second text-encoder to its architecture. Use the [train_dreambooth_lora_sdxl.py](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/train_dreambooth_lora_sdxl.py) script to train a SDXL model with LoRA.

The SDXL training script is discussed in more detail in the [SDXL training](sdxl.md) guide.

## Next steps

Congratulations on training your DreamBooth model! To learn more about how to use your new model, the following guide may be helpful:

- Learn how to [load a DreamBooth](../using-diffusers/loading_adapters.md) model for inference if you trained your model with LoRA.
