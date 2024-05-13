# Stable Diffusion text-to-image fine-tuning

The `train_text_to_image.py` script shows how to fine-tune stable diffusion model on your own dataset.

___Note___:

___This script is experimental. The script fine-tunes the whole model and often times the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyperparameters to get the best result on your dataset.___


## Running locally with MindSpore

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/mindspore-lab/mindone
cd mindone
pip install .
```

Then cd in the example folder `examples/diffusers/text_to_image` and run
```bash
pip install -r requirements.txt
```

### OnePiece example

You need to accept the model license before downloading or using the weights. In this example we'll use model version `v1-4`, so you'll need to visit [its card](https://huggingface.co/CompVis/stable-diffusion-v1-4), read the license and tick the checkbox if you agree.

You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

Run the following command to authenticate your token

```bash
huggingface-cli login
```

If you have already cloned the repo, then you won't need to go through these steps.

<br>

#### Hardware

With `gradient_checkpointing` and `mixed_precision` it should be possible to fine tune the model on a single 24GB NPU. For higher `batch_size` and faster training it's better to use NPUs with >30GB memory.

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="YaYaB/onepiece-blip-captions"

python train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-onepiece-model-$(date +%Y%m%d%H%M%S)"
```

To run on your own training files prepare the dataset according to the format required by `datasets`, you can find the instructions for how to do that in this [document](https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder-with-metadata).
If you wish to use custom loading logic, you should modify the script, we have left pointers for that in the training script.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="path_to_your_dataset"

python train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-your-dataset-model-$(date +%Y%m%d%H%M%S)"
```

Once the training is finished the model will be saved in the `output_dir` specified in the command. In this example it's `sd-onepiece-model`. To load the fine-tuned model for inference just pass that path to `StableDiffusionPipeline`

```python
import mindspore as ms
from mindone.diffusers import StableDiffusionPipeline

model_path = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, mindspore_dtype=ms.float16)

image = pipe(prompt="a man in a straw hat")[0][0]
image.save("a-man-in-a-straw-hat.png")
```

Checkpoints only save the unet, so to run inference from a checkpoint, just load the unet

```python
import mindspore as ms
from mindone.diffusers import StableDiffusionPipeline, UNet2DConditionModel

model_path = "path_to_saved_model"
unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-<N>/unet", mindspore_dtype=ms.float16)

pipe = StableDiffusionPipeline.from_pretrained("<initial model>", unet=unet, mindspore_dtype=ms.float16)

image = pipe(prompt="a man in a straw hat")[0][0]
image.save("a-man-in-a-straw-hat.png")
```

#### Training with Min-SNR weighting

We support training with the Min-SNR weighting strategy proposed in [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556) which helps to achieve faster convergence
by rebalancing the loss. In order to use it, one needs to set the `--snr_gamma` argument. The recommended
value when using it is 5.0.

You can find [this project on Weights and Biases](https://wandb.ai/sayakpaul/text2image-finetune-minsnr) that compares the loss surfaces of the following setups:

* Training without the Min-SNR weighting strategy
* Training with the Min-SNR weighting strategy (`snr_gamma` set to 5.0)
* Training with the Min-SNR weighting strategy (`snr_gamma` set to 1.0)

For our small OnePiece dataset, the effects of Min-SNR weighting strategy might not appear to be pronounced, but for larger datasets, we believe the effects will be more pronounced.

Also, note that in this example, we either predict `epsilon` (i.e., the noise) or the `v_prediction`. For both of these cases, the formulation of the Min-SNR weighting strategy that we have used holds.


## Stable Diffusion XL

* We support fine-tuning the UNet shipped in [Stable Diffusion XL](https://huggingface.co/papers/2307.01952) via the `train_text_to_image_sdxl.py` script. Please refer to the docs [here](./README_sdxl.md).
