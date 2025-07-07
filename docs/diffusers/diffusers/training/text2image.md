<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Text-to-image

!!! warning

    The text-to-image script is experimental, and it's easy to overfit and run into issues like catastrophic forgetting. Try exploring different hyperparameters to get the best results on your dataset.

Text-to-image models like Stable Diffusion are conditioned to generate images given a text prompt.

Training a model can be taxing on your hardware, but if you enable `gradient_checkpointing` and `mixed_precision`, it can reduce video memory usage. You can reduce your memory footprint by enabling memory-efficient attention with [xFormers](../optimization/xformers.md).

This guide will explore the [train_text_to_image.py](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/train_text_to_image.py) training script to help you become familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install .
```

Lastly, if you want to train a model on your own dataset, take a look at the [Create a dataset for training](create_dataset.md) guide to learn how to create a dataset that works with the training script.

## Script parameters

!!! tip

    The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/train_text_to_image.py) and let us know if you have any questions or concerns.

The training script provides many parameters to help you customize your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/train_text_to_image.py#L90) function. This function provides default values for each parameter, such as the training batch size and learning rate, but you can also set your own values in the training command if you'd like.

For example, to speedup training with mixed precision using the fp16 format, add the `--mixed_precision` parameter to the training command:

```bash
python train_text_to_image.py \
  --mixed_precision="fp16"
```

Some basic and important parameters include:

- `--pretrained_model_name_or_path`: the name of the model on the Hub or a local path to the pretrained model
- `--dataset_name`: the name of the dataset on the Hub or a local path to the dataset to train on
- `--image_column`: the name of the image column in the dataset to train on
- `--caption_column`: the name of the text column in the dataset to train on
- `--output_dir`: where to save the trained model
- `--push_to_hub`: whether to push the trained model to the Hub
- `--checkpointing_steps`: frequency of saving a checkpoint as the model trains; this is useful if for some reason training is interrupted, you can continue training from that checkpoint by adding `--resume_from_checkpoint` to your training command

### Min-SNR weighting

The [Min-SNR](https://huggingface.co/papers/2303.09556) weighting strategy can help with training by rebalancing the loss to achieve faster convergence. The training script supports predicting `epsilon` (noise) or `v_prediction`, but Min-SNR is compatible with both prediction types.

Add the `--snr_gamma` parameter and set it to the recommended value of 5.0:

```bash
python train_text_to_image.py \
  --snr_gamma=5.0
```

You can compare the loss surfaces for different `snr_gamma` values in this [Weights and Biases](https://wandb.ai/sayakpaul/text2image-finetune-minsnr) report. For smaller datasets, the effects of Min-SNR may not be as obvious compared to larger datasets.

## Training script

The dataset preprocessing code and training loop are found in the [`main()`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/train_text_to_image.py#L411) function. If you need to adapt the training script, this is where you'll need to make your changes.

The `train_text_to_image` script starts by [loading a scheduler](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/train_text_to_image.py#L437) and tokenizer. You can choose to use a different scheduler here if you want:

```py
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)
```

Then the script [loads the UNet](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/train_text_to_image.py#L437) model:

```py
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
)
unet.register_to_config(sample_size=args.resolution // (2 ** (len(vae.config.block_out_channels) - 1)))
```

Next, the text and image columns of the dataset need to be preprocessed. The [`tokenize_captions`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/train_text_to_image.py#L548) function handles tokenizing the inputs, and the [`train_transforms`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/train_text_to_image.py#L566) function specifies the type of transforms to apply to the image. Both of these functions are bundled into `preprocess_train`:

```py
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image)[0] for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples
```

Lastly, the [training loop](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/train_text_to_image.py#L751) handles everything else. It encodes images into latent space, adds noise to the latents, computes the text embeddings to condition on, updates the model parameters, and saves and pushes the model to the Hub. If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline.md) tutorial which breaks down the basic pattern of the denoising process.

## Launch the script

Once you've made all your changes or you're okay with the default configuration, you're ready to launch the training script! 🚀

Let's train on the [Naruto BLIP captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) dataset to generate your own Naruto characters. Set the environment variables `MODEL_NAME` and `dataset_name` to the model and the dataset (either from the Hub or a local path).

!!! tip

    To train on a local dataset, set the `TRAIN_DIR` and `OUTPUT_DIR` environment variables to the path of the dataset and where to save the model to.

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export dataset_name="lambdalabs/naruto-blip-captions"

python train_text_to_image.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model" \
  --push_to_hub
```

Once training is complete, you can use your newly trained model for inference:

```py
from mindone.diffusers import StableDiffusionPipeline
import mindspore as ms

pipeline = StableDiffusionPipeline.from_pretrained("path/to/saved_model", mindspore_dtype=ms.float16, use_safetensors=True)

image = pipeline(prompt="yoda")[0][0]
image.save("yoda-naruto.png")
```

## Next steps

Congratulations on training your own text-to-image model! To learn more about how to use your new model, the following guides may be helpful:

- Learn how to [load LoRA weights](../using-diffusers/loading_adapters.md#LoRA) for inference if you trained your model with LoRA.
- Learn more about how certain parameters like guidance scale or techniques such as prompt weighting can help you control inference in the [Text-to-image](../using-diffusers/conditional_image_generation.md) task guide.
