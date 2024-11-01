<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Unconditional image generation

Unconditional image generation models are not conditioned on text or images during training. It only generates images that resemble its training data distribution.

This guide will explore the [train_unconditional.py](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/unconditional_image_generation/train_unconditional.py) training script to help you become familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install .
```

Lastly, if you want to train a model on your own dataset, take a look at the [Create a dataset for training](create_dataset.md) guide to learn how to create a dataset that works with the training script.

## Script parameters

!!! tip

    The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/unconditional_image_generation/train_unconditional.py) and let us know if you have any questions or concerns.

The training script provides many parameters to help you customize your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/unconditional_image_generation/train_unconditional.py#L26) function. It provides default values for each parameter, such as the training batch size and learning rate, but you can also set your own values in the training command if you'd like.

For example, to speedup training with mixed precision using the bf16 format, add the `--mixed_precision` parameter to the training command:

```bash
python train_unconditional.py \
  --mixed_precision="bf16"
```

Some basic and important parameters to specify include:

- `--dataset_name`: the name of the dataset on the Hub or a local path to the dataset to train on
- `--output_dir`: where to save the trained model
- `--push_to_hub`: whether to push the trained model to the Hub
- `--checkpointing_steps`: frequency of saving a checkpoint as the model trains; this is useful if training is interrupted, you can continue training from that checkpoint by adding `--resume_from_checkpoint` to your training command

Bring your dataset, and let the training script handle everything else!

## Training script

The code for preprocessing the dataset and the training loop is found in the [`main()`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/unconditional_image_generation/train_unconditional.py#L250) function. If you need to adapt the training script, this is where you'll need to make your changes.

The `train_unconditional` script [initializes a `UNet2DModel`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/unconditional_image_generation/train_unconditional.py#L271) if you don't provide a model configuration. You can configure the UNet here if you'd like:

```py
model = UNet2DModel(
    sample_size=args.resolution,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
```

Next, the script initializes a [scheduler](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/unconditional_image_generation/train_unconditional.py#L309) and [optimizer](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/unconditional_image_generation/train_unconditional.py#L385):

```py
# Initialize the scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=args.ddpm_num_steps,
    beta_schedule=args.ddpm_beta_schedule,
    prediction_type=args.prediction_type,
)


# Initialize the optimizer
optimizer = nn.AdamWeightDecay(
    unet.trainable_params(),
    learning_rate=lr_scheduler,
    beta1=args.adam_beta1,
    beta2=args.adam_beta2,
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

Then it [loads a dataset](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/unconditional_image_generation/train_unconditional.py#L331) and you can specify how to [preprocess](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/unconditional_image_generation/train_unconditional.py#L335) it:

```py
dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

augmentations = transforms.Compose(
    [
        vision.Resize(args.resolution, interpolation=vision.Inter.BILINEAR),
        vision.CenterCrop(args.resolution) if args.center_crop else vision.RandomCrop(args.resolution),
        vision.RandomHorizontalFlip() if args.random_flip else lambda x: x,
        vision.ToTensor(),
        vision.Normalize([0.5], [0.5], is_hwc=False),
    ]
)
```

Finally, the [training loop](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/unconditional_image_generation/train_unconditional.py#L471) handles everything else such as adding noise to the images, predicting the noise residual, calculating the loss, saving checkpoints at specified steps, and saving and pushing the model to the Hub. If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline.md) tutorial which breaks down the basic pattern of the denoising process.

## Launch the script

Once you've made all your changes or you're okay with the default configuration, you're ready to launch the training script! 🚀

```bash
python train_unconditional.py \
  --dataset_name="huggan/flowers-102-categories" \
  --output_dir="ddpm-ema-flowers-64" \
  --mixed_precision="fp16" \
  --push_to_hub
```

The training script creates and saves a checkpoint file in your repository. Now you can load and use your trained model for inference:

```py
from mindone.diffusers import DiffusionPipeline
import mindspore as ms

pipeline = DiffusionPipeline.from_pretrained("anton-l/ddpm-butterflies-128")
image = pipeline()[0][0]
```
