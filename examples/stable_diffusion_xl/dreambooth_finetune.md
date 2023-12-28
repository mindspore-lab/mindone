# DreamBooth finetune for Stable Diffusion XL (SDXL)

[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)

DreamBooth is a method for personalizing text-to-image diffusion models, with just a few images (3~5) of a subject and its name as a Unique Identifier. During fine-tuning, a class-specific prior-preservation loss is applied in parallel, which leverages the semantic prior that the model has on the class and encourages output diversity.

For example, we have 5 images of a specific [dog](https://github.com/google/dreambooth/tree/main/dataset/dog) belonging to the prompt "a sks dog" for fine-tuning, where "sks" is a Unique Identifier. In parallel, images of general dogs, which are the class images in a text prompt "a dog", are inputted, so that the models will not forget other dogs' look.

The `train_dreambooth.py` script implements DreamBooth finetune for SDXL based on MindSpore and Ascend platforms.

**Note**: now we only allow DreamBooth fine-tuning of SDXL UNet via [LoRA](https://arxiv.org/abs/2106.09685) .

## Preparation

#### Dependency

Make sure the following frameworks are installed.

- mindspore 2.1.0 (Ascend 910) / mindspore 2.2.1 (Ascend 910*)
- openmpi 4.0.3 (for distributed mode)

Enter the `example/stable_diffusion_xl` folder and run

```shell
pip install -r requirement.txt
```

#### Pretrained models

Download the official pre-train weights from huggingface, convert the weights from `.safetensors` format to Mindspore `.ckpt` format, and put them to `./checkpoints/` folder. Please refer to SDXL [weight_convertion.md](./weight_convertion.md) for detailed steps.

#### Finetuning Dataset Preparation

The finetuning dataset should contain 3-5 images from the same subject in the same folder.

```text
dir
├── img1.jpg
├── img2.jpg
├── img3.jpg
├── img4.jpg
└── img5.jpg
```

You can find images of different classes in [Google/DreamBooth](https://github.com/google/dreambooth/tree/main). Here we use two examples, [dog](https://github.com/google/dreambooth/tree/main/dataset/dog) and [dog6](https://github.com/google/dreambooth/tree/main/dataset/dog6). They are shown as,

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/961bdff6-f565-4cf2-85ce-e59c6ed547f3" width=800 />
</p>
<p align="center">
  <em> Figure 1. dog example: the five images from the subject dog for finetuning. </em>
</p>

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/a5bef2fc-b613-46de-8021-3e489dd663a1" width=800 />
</p>
<p align="center">
  <em> Figure 2. dog6 example: the five images from the subject dog for finetuning. </em>
</p>


## Finetuning

Before running the fintune scripts `train_dreambooth.py`, please specify the arguments that might vary from users.

* `--instance_data_path=/path/to/finetuning_data `
* `--class_data_path=/path/to/class_image `
* `--weight=/path/to/pretrained_model`
* `--save_path=/path/to/save_models`

Modify other arguments in the shell when running the command or the hyper-parameters in the config file `sd_xl_base_finetune_dreambooth_lora_910*.yaml` if needed.

Run with multiple NPUs (for example, 4) training using :

```shell
mpirun --allow-run-as-root -n 4 python train_dreambooth.py \
  --config configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --instance_data_path /path/to/finetuning_data \
  --instance_prompt "A photo of a sks dog" \
  --class_data_path /path/to/class_image \
  --class_prompt "A photo of a dog" \
  --ms_mode 0 \
  --save_ckpt_interval 500 \
  --is_parallel True \
  --device_target Ascend
```

Launch a standalone training using:

```shell
python train_dreambooth.py \
  --config configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --instance_data_path /path/to/finetuning_data \
  --instance_prompt "A photo of a sks dog" \
  --class_data_path /path/to/class_image \
  --class_prompt "A photo of a dog" \
  --gradient_accumulation_steps 4 \
  --ms_mode 0 \
  --save_ckpt_interval 500 \
  --device_target Ascend
```

Our implementation is trained with prior-preservation loss, which avoids overfitting and language drift. We first generate images using the pertained model with a class prompt, and input those data in parallel with our data during finetuning. The `num_class_images` in the arguments of `train_dreambooth.py`  specifies the number of class images for prior-preservation. If not enough images are present in `class_image_path`, additional images will be sampled with `class_prompt`. And you would need to relaunch the training using the command above when sampling is finished. It takes about 25 minutes to sample 50 class images.

## Inference

Notice that the training command above gets finetuned lora weights in the specified `save_path`. Now we could use the inference command to generate images on a given prompt. Assume that the pretrained ckpt path is `checkpoints/sd_xl_base_1.0_ms.ckpt` and the trained lora ckpt path is `runs/SDXL_base_1.0_1000_lora.ckpt`, examples of inference command are as below.

* (Recommend) Run with interactive visualization.

  Replace the path of weights and yaml file at the constant `VERSION2SPECS`  in `demo/sampling.py`  , specify the prompt in `__main__` and run:

  ```shell
  # (recommend) run with streamlit
  export MS_PYNATIVE_GE=1
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
  streamlit run demo/sampling.py --server.port <your_port>
  ```

* Run with another command:

  ```shell
  # run with other commands
  export MS_PYNATIVE_GE=1
  python demo/sampling_without_streamlit.py \
    --task txt2img \
    --config configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml \
    --weight checkpoints/sd_xl_base_1.0_ms.ckpt,runs/SDXL_base_1.0_1000_lora.ckpt \
    --prompt "a sks dog swimming in a pool" \
    --device_target Ascend
  ```

The two weights (the pre-trained weight and the finetuned lora weight) for the keyword `weight` are separated by a comma without space.

Examples of generated images with the DreamBooth model using different prompts are shown below.

The [dog](https://github.com/google/dreambooth/tree/main/dataset/dog) example finetuning results,

* "A photo of a sks dog swimming in a pool"

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/0ddf4ce1-4177-44c0-84bd-2b15c0e2f6f4" width=700 />



The [dog6](https://github.com/google/dreambooth/tree/main/dataset/dog6) example finetuning results,

* "A photo of a sks dog in a bucket"

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/5144b904-329c-4d83-aa4b-c2f4ecd60ea0" width=700 />



* "A photo of a sks dog in a doghouse"

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/6b2a6656-10a0-4d9d-8542-a9fa0527bc8a" width=700 />
