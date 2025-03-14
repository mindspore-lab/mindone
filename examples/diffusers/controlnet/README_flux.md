# ControlNet training example for FLUX

The `train_controlnet_flux.py` script shows how to implement the ControlNet training procedure and adapt it for [FLUX](https://github.com/black-forest-labs/flux).


> [!NOTE]
> **Memory consumption**
>
> Flux can be quite expensive to run on consumer hardware devices and as a result, ControlNet training of it comes with higher memory requirements than usual.

> **Gated access**
>
> As the model is gated, before using it with diffusers you first need to go to the [FLUX.1 [dev] Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.1-dev), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in: `huggingface-cli login`


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


## Custom Datasets

We support dataset formats:
The original dataset is hosted in the [ControlNet repo](https://huggingface.co/lllyasviel/ControlNet/blob/main/training/fill50k.zip). We re-uploaded it to be compatible with `datasets` [here](https://huggingface.co/datasets/fusing/fill50k). Note that `datasets` handles dataloading within the training script. To use our example, add `--dataset_name=fusing/fill50k \` to the script and remove line `--jsonl_for_train` mentioned below.


We also support importing data from jsonl(xxx.jsonl),using `--jsonl_for_train` to enable it, here is a brief example of jsonl files:
```sh
{"image": "xxx", "text": "xxx", "conditioning_image": "xxx"}
{"image": "xxx", "text": "xxx", "conditioning_image": "xxx"}
```

## Training

Our experiments were conducted on a single 64GB 910* NPU.

We can define the num_layers, num_single_layers, which determines the size of the control.


```bash
python train_controlnet_flux.py \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
    --dataset_name=fusing/fill50k \
    --conditioning_image_column=conditioning_image \
    --image_column=image \
    --caption_column=text \
    --dataloader_num_workers=8 \
    --output_dir="path_to_save_model" \
    --mixed_precision="bf16" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=15000 \
    --checkpointing_steps=200 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --num_double_layers=4 \
    --num_single_layers=0 \
    --seed=42
```

To better track our training experiments, you can use the following flags in the command above:

* `validation_image`, `validation_prompt`, and `validation_steps` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.


### Inference

Once training is done, we can perform inference like so:

```python
import mindspore
from mindone.diffusers.utils import load_image
from mindone.diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from mindone.diffusers.models.controlnet_flux import FluxControlNetModel

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = "path_to_save_model" # 'promeai/FLUX.1-controlnet-lineart-promeai'
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, mindspore_dtype=mindspore.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    mindspore_dtype=mindspore.bfloat16
)

control_image = load_image("https://huggingface.co/promeai/FLUX.1-controlnet-lineart-promeai/resolve/main/images/example-control.jpg")resize((1024, 1024))
prompt = "cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black gold leaf pattern dress and a white apron mouth open holding a fancy black forest cake with candles on top in the kitchen of an old dark Victorian mansion lit by candlelight with a bright window to the foggy forest and very expensive stuff everywhere"

image = pipe(
    prompt,
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=28,
    guidance_scale=3.5,
)[0][0]
image.save("./output.png")
```

### Apply ZeRO3


The training script supports [Zero Redundancy Optimizer (ZeRO)](https://arxiv.org/pdf/1910.02054.pdf) from stage 1 to 3. You could enable ZeRO3 training by setting `--zero_stage=3` and `--distributed`.

Here is an example of of training 512 resolution on 4 NPUs with zero3.

```bash
export OUTPUT_DIR = 'path_to_output"
msrun --worker_num=4 --local_worker_num=4 --log_dir=$OUTPUT_DIR train_controlnet_flux.py \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
    --dataset_name=fusing/fill50k \
    --conditioning_image_column=conditioning_image \
    --image_column=image \
    --caption_column=text \
    --dataloader_num_workers=8 \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="bf16" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=15000 \
    --checkpointing_steps=200 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --num_double_layers=4 \
    --num_single_layers=0 \
    --seed=42 \
    --zero_stage=3 \
    --distributed
```

Refer to the [tutorial](https://github.com/mindspore-lab/mindone/blob/master/docs/tools/zero.md) of using Zero redundancy optimizer(ZeRO) on MindONE if needed.
