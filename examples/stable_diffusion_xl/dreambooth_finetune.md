# DreamBooth finetune for Stable Diffusion XL (SDXL)

[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)

DreamBooth is a method for personalizing text-to-image diffusion models, with just a few images (3~5) of a subject and its name as a Unique Identifier. During fine-tuning, a class-specific prior-preservation loss is applied in parallel, which leverages the semantic prior that the model has on the class and encourages output diversity.

For example, we have 5 images of a specific [dog](https://github.com/google/dreambooth/tree/main/dataset/dog) belonging to the prompt "a sks dog" for fine-tuning, where "sks" is a Unique Identifier. In parallel, images of general dogs, which are the class images in a text prompt "a dog", are inputted, so that the models will not forget other dogs' look. 

The `train_dreambooth.py` script implements DreamBooth finetune for SDXL based on MindSpore and Ascend platforms. 

**Note**: For now we only allow DreamBooth fine-tuning of SDXL UNet via [LoRA](https://arxiv.org/abs/2106.09685) . 

## Preparation

#### Dependency

Make sure the following frameworks are installed.

- mindspore 2.1.0

Enter the `example/stable_diffusion_xl` folder and run

```shell l
pip install -r requirement.txt
```

#### Pretrained models

Download the official pre-train weights from huggingface, convert the weights from `.safetensors` format to Mindspore `.ckpt` format, and put them to `./checkpoints/` folder. Please refer to SDXL [GETTING_STARTED.md](https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_xl/GETTING_STARTED.md#convert-pretrained-checkpoint) for detailed steps.

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

You can find images of different classes in [Google/DreamBooth](https://github.com/google/dreambooth/tree/main). Here we use the [dog](https://github.com/google/dreambooth/tree/main/dataset/dog) example. They are shown as,

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/961bdff6-f565-4cf2-85ce-e59c6ed547f3" width=800 />
</p>
<p align="center">
  <em> Figure 1. The five images from the subject dog for finetuning. </em>
</p>

## Finetuning

Before running the fintune scripts `train_dreambooth.py`, please specify the arguments that might vary from users.

* `--instance_data_path=/path/to/finetuning_data `
* `--class_data_path=/path/to/class_image `
* `--weight=/path/to/pretrained_model`
* `--save_path=/path/to/save_models`

Modify other arguments in the shell when running the command or the hyper-parameters in the config file `sd_xl_base_finetune_dreambooth_lora.yaml` if needed.

Launch a standalone training using: 

```shell
python train_dreambooth.py \
  --config configs/training/sd_xl_base_finetune_dreambooth_lora.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --instance_data_path /path/to/finetuning_data \
  --instance_prompt "a photo of a sks dog" \
  --class_data_path /path/to/class_image \
  --class_prompt "a photo of a dog" \
  --device_target Ascend
```

Our implement is trained with prior-preservation loss, which avoids overfitting and language drift. We first generate images using the pertained model with a class prompt, and input those data in parallel with our data during finetuning. The `num_class_images` in the arguments of `train_dreambooth.py`  specifies the number of class images for prior-preservation. If there are not enough images present in `class_image_path`, additional images will be sampled with `class_prompt`. And you would need to relaunch the training using the command above when sampling is finished. It takes about 45 minutes to sample 50 class images. 

## Inference

Notice that the training command above gets finetuned lora weights in the specified `save_path`. Now we could use the inference command to generate images on a given prompt. Assume that the path of the trained lora weight is `output/SDXL_base_1.0_12000_lora.ckpt`, an example inference command is as

```shell
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_dreambooth_lora.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt,output/SDXL_base_1.0_12000_lora.ckpt \
  --prompt "a sks dog swimming in a pool" \
  --device_target Ascend
```

The two weights (the pre-trained weight and the finetuned lora weight) for the keyword `weight` are separated by a comma without space.

Examples of generated images with the DreamBooth model using prompts are shown as below,

* "a sks dog swimming in a pool"

* "a sks dog in a bucket"

* "a sks dog in Van Gogh painting style"

* "a sks dog under a cherry blossom tree"

* "a sks dog playing on the hiil, cold color palette, muted color, detailed"

* "a sks dog playing on the hiil, warm color palette, muted color, detailed"

  

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/bd61f3cd-dcf5-44ec-9ba7-3920004293cb" width=700 />
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/55cc0d71-7760-4368-93d3-b36a0aaa4004" width=700 />
   <img src="https://github.com/mindspore-lab/mindone/assets/33061146/b036a3b3-a2e0-4e22-bcce-d97ddbfc69e7" width=700 />
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/86a9ad6e-1cd0-4880-9308-6a5b97769cfa" width=700 />
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/405f0adb-bf18-4a7e-9cdf-fcf0754e10ee" width=700 />
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/7843ac3e-2254-45cc-998d-ee6279412052" width=700 />
  </p>
<p align="center">
  <em> Figure 2. Some inference examples of a "sks" dog, without refiner. </em>
</p>


​    

  



  
