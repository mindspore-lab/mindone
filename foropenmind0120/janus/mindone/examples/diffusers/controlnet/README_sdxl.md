# ControlNet training example for Stable Diffusion XL (SDXL)

The `train_controlnet_sdxl.py` script shows how to implement the ControlNet training procedure and adapt it for [Stable Diffusion XL](https://huggingface.co/papers/2307.01952).

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


## Circle filling dataset

The original dataset is hosted in the [ControlNet repo](https://huggingface.co/lllyasviel/ControlNet/blob/main/training/fill50k.zip). We re-uploaded it to be compatible with `datasets` [here](https://huggingface.co/datasets/fusing/fill50k). Note that `datasets` handles dataloading within the training script.

## Training

Our training examples use two test conditioning images. They can be downloaded by running

```sh
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png

wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

```bash
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="path to save model"

python train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=60000 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --validation_steps=100 \
 --train_batch_size=1 \
 --seed=42
```

To better track our training experiments, we're using the following flags in the command above:

* `validation_image`, `validation_prompt`, and `validation_steps` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

### Inference

Once training is done, we can perform inference like so:

```python
from mindone.diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from mindone.diffusers.utils import load_image
import mindspore as ms
import numpy as np

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_path = "path to controlnet"

controlnet = ControlNetModel.from_pretrained(controlnet_path, mindspore_dtype=ms.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, mindspore_dtype=ms.float16
)

# speed up diffusion process with faster scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

control_image = load_image("./conditioning_image_1.png").resize((1024, 1024))
prompt = "pale golden rod circle with old lace background"

# generate image
generator = np.random.Generator(np.random.PCG64(0))
image = pipe(
    prompt, num_inference_steps=20, generator=generator, image=control_image
)[0][0]
image.save("./output.png")
```

## Notes

### Specifying a better VAE

SDXL's VAE is known to suffer from numerical instability issues. This is why we also expose a CLI argument namely `--pretrained_vae_model_name_or_path` that lets you specify the location of an alternative VAE (such as [`madebyollin/sdxl-vae-fp16-fix`](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)).

If you're using this VAE during training, you need to ensure you're using it during inference too. You do so by:

```diff
+ vae = AutoencoderKL.from_pretrained(vae_path_or_repo_id, torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16,
+   vae=vae,
)
