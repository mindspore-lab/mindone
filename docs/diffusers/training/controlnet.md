<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNet

[ControlNet](https://hf.co/papers/2302.05543) models are adapters trained on top of another pretrained model. It allows for a greater degree of control over image generation by conditioning the model with an additional input image. The input image can be a canny edge, depth map, human pose, and many more.

If you want to reduce memory footprint, you should try enabling the `gradient_checkpointing`, `gradient_accumulation_steps`, and `mixed_precision` parameters in the training command. You can also reduce your memory footprint by using memory-efficient attention with [xFormers](../optimization/xformers.md).

This guide will explore the [train_controlnet.py](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/controlnet/train_controlnet.py) training script to help you become familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install .
```

Lastly, if you want to train a model on your own dataset, take a look at the [Create a dataset for training](create_dataset.md) guide to learn how to create a dataset that works with the training script.

!!! tip

    The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/controlnet/train_controlnet.py) and let us know if you have any questions or concerns.

## Script parameters

The training script provides many parameters to help you customize your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/controlnet/train_controlnet.py#L147) function. This function provides default values for each parameter, such as the training batch size and learning rate, but you can also set your own values in the training command if you'd like.

For example, to speedup training with mixed precision using the fp16 format, add the `--mixed_precision` parameter to the training command:

```bash
python train_controlnet.py \
  --mixed_precision="fp16"
```

Many of the basic and important parameters are described in the [Text-to-image](text2image.md#script-parameters) training guide, so this guide just focuses on the relevant parameters for ControlNet:

- `--max_train_samples`: the number of training samples; this can be lowered for faster training, but if you want to stream really large datasets, you'll need to include this parameter and the `--streaming` parameter in your training command
- `--gradient_accumulation_steps`: number of update steps to accumulate before the backward pass; this allows you to train with a bigger batch size than your NPU memory can typically handle

## Training script

As with the script parameters, a general walkthrough of the training script is provided in the [Text-to-image](text2image.md#training-script) training guide. Instead, this guide takes a look at the relevant parts of the ControlNet script.

The training script has a [`make_train_dataset`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/controlnet/train_controlnet.py#L510) function for preprocessing the dataset with image transforms and caption tokenization. You'll see that in addition to the usual caption tokenization and image transforms, the script also includes transforms for the conditioning image.

```py
conditioning_image_transforms = transforms.Compose(
    [
        vision.Resize(args.resolution, interpolation=vision.Inter.BILINEAR),
        vision.CenterCrop(args.resolution),
        vision.ToTensor(),
    ]
)
```

Within the [`main()`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/controlnet/train_controlnet.py#L638) function, you'll find the code for loading the tokenizer, text encoder, scheduler and models. This is also where the ControlNet model is loaded either from existing weights or randomly initialized from a UNet:

```py
if args.controlnet_model_name_or_path:
    logger.info("Loading existing controlnet weights")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
else:
    logger.info("Initializing controlnet weights from unet")
    controlnet = ControlNetModel.from_unet(unet)
```

The [optimizer](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/controlnet/train_controlnet.py#L776) is set up to update the ControlNet parameters:

```py
params_to_optimize = controlnet.trainable_params()
optimizer = nn.AdamWeightDecay(
    params_to_optimize,
    learning_rate=lr_scheduler,
    beta1=args.adam_beta1,
    beta2=args.adam_beta2,
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

Finally, in the [training loop](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/controlnet/train_controlnet.py#L846), the conditioning text embeddings and image are passed to the down and mid-blocks of the ControlNet model:

```py
encoder_hidden_states = self.text_encoder(input_ids, return_dict=False)[0]
controlnet_image = conditioning_pixel_values.to(dtype=self.weight_dtype)

down_block_res_samples, mid_block_res_sample = self.controlnet(
    noisy_latents,
    timesteps,
    encoder_hidden_states=encoder_hidden_states,
    controlnet_cond=controlnet_image,
    return_dict=False,
)
```

If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline.md) tutorial which breaks down the basic pattern of the denoising process.

## Launch the script

Now you're ready to launch the training script! 🚀

This guide uses the [fusing/fill50k](https://huggingface.co/datasets/fusing/fill50k) dataset, but remember, you can create and use your own dataset if you want (see the [Create a dataset for training](create_dataset.md) guide).

Set the environment variable `MODEL_NAME` to a model id on the Hub or a path to a local model and `OUTPUT_DIR` to where you want to save the model.

Download the following images to condition your training with:

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

Then launch the script!

```bash
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="path/to/save/model"

python train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --push_to_hub
```

Once training is complete, you can use your newly trained model for inference!

```py
from mindone.diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from mindone.diffusers.utils import load_image
import mindspore as ms
import numpy as np

controlnet = ControlNetModel.from_pretrained("path/to/controlnet", mindspore_dtype=ms.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "path/to/base/model", controlnet=controlnet, mindspore_dtype=ms.float16
)

control_image = load_image("./conditioning_image_1.png")
prompt = "pale golden rod circle with old lace background"

generator = np.random.Generator(np.random.PCG64(0))
image = pipeline(prompt, num_inference_steps=20, generator=generator, image=control_image)[0][0]
image.save("./output.png")
```

## Stable Diffusion XL

Stable Diffusion XL (SDXL) is a powerful text-to-image model that generates high-resolution images, and it adds a second text-encoder to its architecture. Use the [`train_controlnet_sdxl.py`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/controlnet/train_controlnet_sdxl.py) script to train a ControlNet adapter for the SDXL model.

The SDXL training script is discussed in more detail in the [SDXL training](sdxl.md) guide.

## Next steps

Congratulations on training your own ControlNet! To learn more about how to use your new model, the following guides may be helpful:

- Learn how to [use a ControlNet](../using-diffusers/controlnet.md) for inference on a variety of tasks.
