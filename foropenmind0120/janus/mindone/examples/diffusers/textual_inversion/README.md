## Textual Inversion fine-tuning example

[Textual inversion](https://arxiv.org/abs/2208.01618) is a method to personalize text2image models like stable diffusion on your own images using just 3-5 examples.
The `textual_inversion.py` script shows how to implement the training procedure and adapt it for stable diffusion.


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


### Cat toy example

First, let's login so that we can upload the checkpoint to the Hub during training:

```bash
huggingface-cli login
```

Now let's get our dataset. For this example we will use some cat images: https://huggingface.co/datasets/diffusers/cat_toy_example .

Let's first download it locally:

```py
from huggingface_hub import snapshot_download

local_dir = "./cat"
snapshot_download("diffusers/cat_toy_example", local_dir=local_dir, repo_type="dataset", ignore_patterns=".gitattributes")
```

This will be our training data.
Now we can launch the training using:

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

**___Note: Please follow the [README_sdxl.md](./README_sdxl.md) if you are using the [stable-diffusion-xl](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).___**

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./cat"

python textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" \
  --initializer_token="toy" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_cat"
```

**Note**: As described in [the official paper](https://arxiv.org/abs/2208.01618)
only one embedding vector is used for the placeholder token, *e.g.* `"<cat-toy>"`.
However, one can also add multiple embedding vectors for the placeholder token
to increase the number of fine-tuneable parameters. This can help the model to learn
more complex details. To use multiple embedding vectors, you should define `--num_vectors`
to a number larger than one, *e.g.*:
```bash
--num_vectors 5
```

The saved textual inversion vectors will then be larger in size compared to the default case.

### Inference

Once you have trained a model using above command, the inference can be done simply using the `StableDiffusionPipeline`. Make sure to include the `placeholder_token` in your prompt.

```python
import mindspore as ms
from mindone.diffusers import StableDiffusionPipeline

model_id = "path-to-your-trained-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, mindspore_dtype=ms.float16)

prompt = "A <cat-toy> backpack"

image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5)[0][0]

image.save("cat-backpack.png")
```
