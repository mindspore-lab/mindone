# Stable Diffusion unCLIP Finetuning

## Introduction
unCLIP is the approach behind OpenAI's DALL·E 2, trained to invert CLIP image embeddings. This method finetunes SD to accept a CLIP ViT-L/14 image embedding in addition to the text encodings. This means that the model can be used to produce image variations with or without text input.

## Get Started

### Preparation

#### Dependency

Please make sure the following frameworks are installed.

- mindspore >= 1.9  [[install](https://www.mindspore.cn/install)] (2.0 is recommended for the best performance.)
- python >= 3.7
- openmpi 4.0.3 (for distributed training/evaluation)  [[install](https://www.open-mpi.org/software/ompi/v4.0/)]

Install the dependent packages by running:
```shell
pip install -r requirements.txt
```

#### Pretrained Models

Please download the pretrained checkpoint [SD2.0-v-pred checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_768_v-e12e3a9b.ckpt) and [SD-unclip-l checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd21-unclip-l-baa7c8b5.ckpt) and put it under `models/` folder, and run
```bash
python tools/model_conversion/unclip/prepare_unclip_train.py
```
to combine the parameters from two checkpoints into single one. The combined checkpoint is then saved as `models/sd_v2_v_embedder.ckpt`.

#### Text-image Dataset Preparation

The text-image pair dataset for finetuning should follow the file structure below

```text
dir
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── img_txt.csv
```

img_txt.csv is the annotation file in the following format
```text
dir,text
img1.jpg,a cartoon character with a potted plant on his head
img2.jpg,a drawing of a green pokemon with red eyes
img3.jpg,a red and white ball with an angry look on its face
```

For convenience, we have prepared one public text-image dataset obeying the above format.

- [pokemon-blip-caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 833 pokemon-style images with BLIP-generated captions.

To use it, please download `pokemon_blip.zip` from the [openi dataset website](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets). Then unzip them on your local directory, e.g. `./datasets/pokemon_blip`.

### unCLIP Finetune

We will use the `train_unclip_image_variation.py` script to train the unclip image variation.
Before running, please modify the following arguments to your local path in the shell or in the config file `train_config_vanilla_v2_vpred_unclip_l.yaml`:

* `--data_path=/path/to/data`
* `--output_path=/path/to/save/output_data`
* `--pretrained_model_path=/path/to/pretrained_model`

Then, execute the script to launch finetuning:

```shell
mpirun -n 4 python train_unclip_image_variation.py \
    --train_config "configs/train/train_config_vanilla_v2_vpred_unclip_l.yaml" \
    --data_path "datasets/pokemon_blip/train" \
    --pretrained_model_path "models/sd_v2_v_embedder.ckpt" \
    --output_path unclip-train
```

> Note: to modify other important hyper-parameters, please refer to training config file `train_config_vanilla_v2_vpred_unclip_l.yaml`.

After training, the checkpoint will be saved in `output/unclip-train/ckpt/sd-600.ckpt` by default.

Below are some arguments that you may want to tune for a better performance on your dataset:

- `train_batch_size`: the number of batch size for training.
- `start_learning_rate` and `end_learning_rate`: the initial and end learning rates for training.
- `epochs`: the number of epochs for training.
- `use_ema`: whether use EMA for model smoothing

For more argument illustration, please run `python train_unclip_image_variation.py -h`.

### Inference

To perform image-to-image generation with the finetuned checkpoint, please prepare a test image and run

```shell
python unclip_image_variation.py \
    --config configs/v2-vpred-inference-unclip-l.yaml \
    --ckpt_path path_of_the_finetune_checkpoint \
    --image_path path_of_the_test_image \
    --prompt "a picture of a unicorn with orange hair"
```

Here are the example results.

| Prompt  | Image Input  | Image Output 1 | Image Output 2 |
|---|---|---|---|
| "a blue jellyfish with red eyes and a red nose"  | <img src="https://github.com/zhtmike/mindone/assets/8342575/064fad04-de12-4910-bceb-22ac48f687a0" width="200"/> | <img src="https://github.com/zhtmike/mindone/assets/8342575/65e2d215-1eb2-4c1a-8131-6a182c9ad357" width="200"/> | <img src="https://github.com/zhtmike/mindone/assets/8342575/4bc91c1a-c72e-464a-843d-fc457c6793f4" width="200"/> |
| "" | <img src="https://github.com/zhtmike/mindone/assets/8342575/064fad04-de12-4910-bceb-22ac48f687a0" width="200"/> | <img src="https://github.com/zhtmike/mindone/assets/8342575/9f579a66-1e7c-4c3f-8fcb-3947ad88a21d" width="200"/> | <img src="https://github.com/zhtmike/mindone/assets/8342575/122c97ca-39ff-4395-b243-aa9b45b95b3c" width="200"/> |
| "a picture of a unicorn with orange hair" | <img src="https://github.com/zhtmike/mindone/assets/8342575/0dc0d98c-4462-4dd1-8a29-59f427124466" width="200"/> | <img src="https://github.com/zhtmike/mindone/assets/8342575/b8e1a149-2ef0-4da0-9c08-972d1ed3f3cc" width="200"/> | <img src="https://github.com/zhtmike/mindone/assets/8342575/b4c18145-d143-4211-b188-54157062c13f" width="200"/> |
| "" | <img src="https://github.com/zhtmike/mindone/assets/8342575/0dc0d98c-4462-4dd1-8a29-59f427124466" width="200"/> | <img src="https://github.com/zhtmike/mindone/assets/8342575/9199dc61-1b65-41b8-b18a-b582ccc0e55d" width="200"/>  | <img src="https://github.com/zhtmike/mindone/assets/8342575/3b31d969-0130-4b9f-8501-df9bda835147" width="200"/>  |
| "a cartoon picture of a blue and white pokemon" | <img src="https://github.com/zhtmike/mindone/assets/8342575/d39d97bf-8116-493c-9a73-d414ad2293e1" width="200"/> | <img src="https://github.com/zhtmike/mindone/assets/8342575/1736be4e-ba56-49e1-ab7f-77c47c96c30b" width="200"/> | <img src="https://github.com/zhtmike/mindone/assets/8342575/ce33be16-eb5a-4e48-9506-77a4d36acc3b" width="200"/> |
| "" | <img src="https://github.com/zhtmike/mindone/assets/8342575/d39d97bf-8116-493c-9a73-d414ad2293e1" width="200"/>  | <img src="https://github.com/zhtmike/mindone/assets/8342575/1524108d-37f8-4133-b830-71dd5ac52351" width="200"/> | <img src="https://github.com/zhtmike/mindone/assets/8342575/b1afc3d3-c5cc-49ae-ad27-bcd149885680" width="200"/>|
