<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Train a diffusion model

Unconditional image generation is a popular application of diffusion models that generates images that look like those in the dataset used for training. Typically, the best results are obtained from finetuning a pretrained model on a specific dataset. You can find many of these checkpoints on the [Hub](https://huggingface.co/search/full-text?q=unconditional-image-generation&type=model), but if you can't find one you like, you can always train your own!

This tutorial will teach you how to train a [`UNet2DModel`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/unet2d/#mindone.diffusers.UNet2DModel) from scratch on a subset of the [Smithsonian Butterflies](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) dataset to generate your own 🦋 butterflies 🦋.

!!! tip

    💡 This training tutorial is based on the [Training with 🧨 Diffusers](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) notebook. For additional details and context about diffusion models like how they work, check out the notebook!


Before you begin, make sure you have 🤗 Datasets installed to load and preprocess image datasets. The following command will also install [TensorBoard](https://www.tensorflow.org/tensorboard) to visualize training metrics (you can also use [Weights & Biases](https://docs.wandb.ai/) to track your training).

```py
# uncomment to install the necessary libraries in Colab
#!pip install mindone[training]
```

We encourage you to share your model with the community, and in order to do that, you'll need to login to your Hugging Face account (create one [here](https://hf.co/join) if you don't already have one!). You can login from a notebook and enter your token when prompted. Make sure your token has the write role.

```python
from huggingface_hub import notebook_login

notebook_login()
```

Or login in from the terminal:

```bash
huggingface-cli login
```

Since the model checkpoints are quite large, install [Git-LFS](https://git-lfs.com/) to version these large files:

```bash
!sudo apt -qq install git-lfs
!git config --global credential.helper store
```

## Training configuration

For convenience, create a `TrainingConfig` class containing the training hyperparameters (feel free to adjust them):

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()
```

## Load the dataset

You can easily load the [Smithsonian Butterflies](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) dataset with the 🤗 Datasets library:

```python
from datasets import load_dataset

config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")
```

!!! tip

    💡 You can find additional datasets from the [HugGan Community Event](https://huggingface.co/huggan) or you can use your own dataset by creating a local [`ImageFolder`](https://huggingface.co/docs/datasets/image_dataset#imagefolder). Set `config.dataset_name` to the repository id of the dataset if it is from the HugGan Community Event, or `imagefolder` if you're using your own images.

🤗 Datasets uses the [`~datasets.Image`] feature to automatically decode the image data and load it as a [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) which we can visualize:

```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["image"]):
    axs[i].imshow(image)
    axs[i].set_axis_off()
fig.show()
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/butterflies_ds.png"/>
</div>

The images are all different sizes though, so you'll need to preprocess them first:

* `Resize` changes the image size to the one defined in `config.image_size`.
* `RandomHorizontalFlip` augments the dataset by randomly mirroring the images.
* `Normalize` is important to rescale the pixel values into a [-1, 1] range, which is what the model expects.

```python
from mindspore.dataset import transforms, vision

preprocess = transforms.Compose(
    [
        vision.Resize((config.image_size, config.image_size)),
        vision.RandomHorizontalFlip(),
        vision.ToTensor(),
        vision.Normalize([0.5], [0.5], is_hwc=False),
    ]
)
```

Use 🤗 Datasets' [`~datasets.Dataset.set_transform`] method to apply the `preprocess` function on the fly during training:

```python
def transform(examples):
    images = [preprocess(image.convert("RGB"))[0] for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)
```

Feel free to visualize the images again to confirm that they've been resized. Now you're ready to wrap the dataset in a [DataLoader](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html) for training!

```python
from mindspore.dataset import GeneratorDataset

class DatasetForMindData:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        idx = idx.item() if isinstance(idx, np.integer) else idx
        return np.array(self.data[idx]["images"], dtype=np.float32)

    def __len__(self):
        return len(self.data)

train_dataloader = GeneratorDataset(DatasetForMindData(dataset), column_names=["images"], shuffle=True).batch(batch_size=config.train_batch_size)
```

## Create a UNet2DModel

Pretrained models in 🧨 Diffusers are easily created from their model class with the parameters you want. For example, to create a [`UNet2DModel`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/unet2d/#mindone.diffusers.UNet2DModel):

```python
from mindone.diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
```

It is often a good idea to quickly check the sample image shape matches the model output shape:

```python
sample_image = mindspore.Tensor(dataset[0]["images"]).unsqueeze(0)
print("Input shape:", sample_image.shape)
Input shape: (1, 3, 128, 128)

print("Output shape:", model(mindspore.Tensor(sample_image), timestep=0)[0].shape)
Output shape: (1, 3, 128, 128)
```

Great! Next, you'll need a scheduler to add some noise to the image.

## Create a scheduler

The scheduler behaves differently depending on whether you're using the model for training or inference. During inference, the scheduler generates image from the noise. During training, the scheduler takes a model output - or a sample - from a specific point in the diffusion process and applies noise to the image according to a *noise schedule* and an *update rule*.

Let's take a look at the [`DDPMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/ddpm/#mindone.diffusers.DDPMScheduler) and use the `add_noise` method to add some random noise to the `sample_image` from before:

```python
import mindspore
from PIL import Image
from mindone.diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = mindspore.ops.randn(sample_image.shape)
timesteps = mindspore.Tensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(mindspore.uint8).numpy()[0])
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/87d5b944-9f1f-40db-b6cd-95c1bf8e1683"/>
</div>

The training objective of the model is to predict the noise added to the image. The loss at this step can be calculated by:

```python
from mindspore import ops

noise_pred = model(noisy_image, timesteps)[0]
loss = ops.mse_loss(noise_pred, noise)
```

## Train the model

By now, you have most of the pieces to start training the model and all that's left is putting everything together.

First, you'll need an optimizer and a learning rate scheduler:

```python
from mindspore import nn
from mindone.diffusers.optimization import get_cosine_schedule_with_warmup

lr_scheduler = get_cosine_schedule_with_warmup(
    config.learning_rate,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=lr_scheduler)
```

Then, you'll need a way to evaluate the model. For evaluation, you can use the [`DDPMPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/unet2d/#mindone.diffusers.DDPMPipeline) to generate a batch of sample images and save it as a grid:

```python
import numpy as np
from mindone.diffusers import DDPMPipeline
from mindone.diffusers.utils import make_image_grid
import os

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=np.random.Generator(np.random.PCG64(config.seed)),
    )[0]

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
```

Now you can wrap all these components together in a training loop with TensorBoard logging, gradient accumulation, and mixed precision training. To upload the model to the Hub, write a function to get your repository name and information and then push it to the Hub.

!!! tip

    💡 The training loop below may look intimidating and long, but it'll be worth it later when you launch your training in just one line of code! If you can't wait and want to start generating images, feel free to copy and run the code below. You can always come back and examine the training loop more closely later, like when you're waiting for your model to finish training. 🤗

```python
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
from mindone.diffusers.training_utils import TrainStep
from mindspore.amp import StaticLossScaler

# Write your train step
class MyTrainStep(TrainStep):
    def __init__(
        self,
        model: nn.Cell,
        optimizer: nn.Optimizer,
        noise_scheduler,
        gradient_accumulation_steps,
        length_of_dataloader,
    ):
        super().__init__(
            model,
            optimizer,
            StaticLossScaler(65536),
            1.0,
            gradient_accumulation_steps,
            gradient_accumulation_kwargs=dict(length_of_dataloader=length_of_dataloader),
        )
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_num_train_timesteps = noise_scheduler.config.num_train_timesteps

    def forward(self, clean_images):
        # Sample noise to add to the images
        noise = ops.randn(clean_images.shape)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = ops.randint(
            0, self.noise_scheduler_num_train_timesteps, (bs,), dtype=mindspore.int64
        )

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
        loss = ops.mse_loss(noise_pred, noise)
        loss = self.scale_loss(loss)
        return loss, noise_pred

is_main_process, is_local_main_process = True, True
train_step = MyTrainStep(model, optimizer, noise_scheduler, config.gradient_accumulation_steps, len(train_dataloader)).set_train()
pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    if is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id

        global_step = 0

        # Now you train the model
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                loss, model_pred = train_step(*batch)

                progress_bar.update(1)
                logs = {"loss": loss.item(), "lr": optimizer.get_lr().numpy().item(), "step": global_step}
                progress_bar.set_postfix(**logs)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if is_main_process:
                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    evaluate(config, epoch, pipeline)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    if config.push_to_hub:
                        upload_folder(
                            repo_id=repo_id,
                            folder_path=config.output_dir,
                            commit_message=f"Epoch {epoch}",
                            ignore_patterns=["step_*", "epoch_*"],
                        )
                    else:
                        pipeline.save_pretrained(config.output_dir)
```

If you want to launch a distributed training, see [tutorial](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/overview.html) from mindspore. And you can get the rank of process by:

```python
from mindspore.communication import get_local_rank, get_rank
rank, local_rank = get_rank(), get_local_rank()
is_main_process, is_local_main_process = rank == 0, local_rank == 0

mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_syntax_level=mindspore.STRICT)
train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
```

Once training is complete, take a look at the final 🦋 images 🦋 generated by your diffusion model!

```python
import glob

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/089f833c-f507-46dc-a948-8dd6b941a0b7"/>
</div>

## Next steps

Unconditional image generation is one example of a task that can be trained. You can explore other tasks and training techniques by visiting the [🧨 Diffusers Training Examples](../training/overview.md) page. Here are some examples of what you can learn:

* [Textual Inversion](../training/text_inversion.md), an algorithm that teaches a model a specific visual concept and integrates it into the generated image.
* [DreamBooth](../training/dreambooth.md), a technique for generating personalized images of a subject given several input images of the subject.
* [Guide](../training/text2image.md) to finetuning a Stable Diffusion model on your own dataset.
* [Guide](../training/lora.md) to using LoRA, a memory-efficient technique for finetuning really large models faster.
