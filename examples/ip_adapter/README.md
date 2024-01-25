# IP-Adapter

## Introduction

This folder contains [IP-Adapter](https://arxiv.org/abs/2308.06721) models implemented with [MindSpore](https://www.mindspore.cn/), reference to [Official Implementation](https://github.com/tencent-ailab/IP-Adapter).

IP-Adapter is an effective and lightweight adapter to achieve image prompt capability for the pre-trained text-to-image diffusion models. An IP-Adapter with only 22M parameters can achieve comparable or even better performance to a fine-tuned image prompt model. IP-Adapter can be generalized not only to other custom models fine-tuned from the same base model, but also to controllable generation using existing controllable tools. Moreover, the image prompt can also work well with the text prompt to accomplish multimodal image generation.

<p align="center"><img width="700" alt="IP-Adapter Architecture" src="https://github.com/zhtmike/mindone/assets/8342575/f332b980-0fb9-4fe6-bc08-f66c33cabc8f"/>
<br><em>Overall IP-Adapter architecture</em></p>

## Dependency

- mindspore 2.2

To install the dependency, please run

```shell
pip install -r requirements.txt
```

## Preparation

### Download Models

You can download the following models before running the inference

- [sd_v1.5-d0ab7146.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt) (required by SD-1.5)
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) (required by SD-1.5)
- [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (required by SD-XL)
- [IP-Adapter weight](https://huggingface.co/h94/IP-Adapter) (required by SD-1.5 & SD-XL)
- [ControlNet](https://huggingface.co/lllyasviel) (optionally required by SD-1.5)
- [ControlNet SDXL](https://huggingface.co/diffusers) (optionally required by SD-XL)

### Converting Models

To convert the model weight into Mindspore `.ckpt` checkpoint format, you can follow the [instructions](../stable_diffusion_xl/GETTING_STARTED.md#convert-pretrained-checkpoint) from the SDXL project to convert `sd_xl_base_1.0.safetensors` into `sd_xl_base_1.0_ms.ckpt`. For SD-1.5, you can directly download the converted checkpoint from [Mindspore FTP website](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt). Once you have the converted checkpoint, please put it under the `checkpoints/` directory. Additionally, please download the checkpoint from [IP-Adapter weight](https://huggingface.co/h94/IP-Adapter) and put it under the same directory.

The final folder structure should look like this:

```text
checkpoints/
├── sd_models
│   ├── IP-Adapter
│   │   ├── image_encoder
│   │   │   └── model.safetensors
│   │   └── ip-adapter_sd15.safetensors
│   ├── sd_v1.5_d0ab7146.ckpt
│   └── sd-vae-ft-mse
│       └── diffusion_pytorch_model.safetensors
└── sdxl_models
    ├── IP-Adapter
    │   ├── image_encoder
    │   │   └── model.safetensors
    │   ├── ip-adapter_sdxl.safetensors
    │   └── ip-adapter_sdxl_vit-h.safetensors
    └── sd_xl_base_1.0_ms.ckpt
```

We also provide scripts to merge the weights from stable diffusion XL / 1.5, image encoder, and IP-Adapter into a single file, which can be loaded by the inference script later. To merge the checkpoint for stable diffusion XL, please run the following command:

```bash
python tools/merge_ckpts_sdxl.py
```

For stable diffusion 1.5, please run the following command:

```bash
python tools/merge_ckpts_sd.py
```

The merged checkpoint `sd_v1.5_ip_adapter.ckpt` and `sd_xl_base_1.0_ms_ip_adapter.ckpt` will be saved under `merged/` directory. For more information on how to use this script, please run `python tools/merge_ckpts_sdxl.py -h` and `python tools/merge_ckpts_sd.py -h`.

## Inference and Examples

### Image Variation (SD-XL)

To run the image variation task on SD-XL, you can use the `image_variation_sdxl.py` script. First, prepare the input image and the converted checkpoint. As an example, you may download the [input image](https://github.com/zhtmike/mindone/assets/8342575/dd266d78-2b4b-4159-ad95-5a6afc138235) and saved it as `assets/woman.png`. Then, run the following command in your terminal:
```bash
python image_variation_sdxl.py --img path_of_the_image --weight path_of_the_ckpt --num_cols 4
```
The `--num_cols` flag denotes the number of output images in a single trial. The output images will be saved under `outputs/demo/SDXL-base-1.0` directory. You can also add a prompt by running the following command:
```bash
python image_variation_sdxl.py --img path_of_the_image --weight path_of_the_ckpt --prompt "your magic prompt" --num_cols 4 --ip_scale 0.6
```
The `--ip_scale` flag controls the amount of influence of the image input. The larger the value, the more the output image will resemble the input image, which has the potential of ignoring the input prompt. Here are some examples (The image on the left is the original input, while the remaining images are the outputs.):

```bash
python image_variation_sdxl.py --img assets/woman.png --num_cols 4
```
<p align="center"><img width="1200" src="https://github.com/zhtmike/mindone/assets/8342575/9c021192-c262-46c1-b9fd-703700395360"/>
<br><em>An example of image variation without a prompt (SD-XL)</em></p>

```bash
python image_variation_sdxl.py --img assets/woman.png --prompt "best quality, high quality, wearing sunglasses on the beach"  --num_cols 4 --ip_scale 0.6 --seed 43
```
<p align="center"><img width="1200" src="https://github.com/zhtmike/mindone/assets/8342575/4ca4ffd6-89b5-4b9f-822b-0fd51e8bc2da"/>
<br><em>An example of image variation with a given prompt (SD-XL)</em></p>

For more information on how to use this script, please run `python image_variation_sdxl.py -h`.

### ControlNet (SD-XL)

To run the ControlNet task on SD-XL, you can use the `controlnet_sdxl.py` script. First, download the SD-XL ControlNet checkpoint (in `.safetensors` format) from (https://huggingface.co/diffusers), save them under `checkpoints/sdxl_models`. Prepare the converted checkpoint by running

```bash
# we use vit-h-14 encoder here instead of vit-bigG/14 for image encoding
python tools/merge_ckpts_sdxl.py \
    --ip_adapter checkpoints/sdxl_models/IP-Adapter/ip-adapter_sdxl_vit-h.safetensors \
    --open_clip_vit checkpoints/sd_models/IP-Adapter/image_encoder/model.safetensors \
    --controlnet path_of_the_controlnet_checkpoint
```

The converted checkpoint `sd_xl_base_1.0_ms_ip_adapter.ckpt` will be saved under `checkpoints/sdxl_models/merged` directory. Then, prepare then input image and the control image, as an example, you may download the [input image](https://github.com/zhtmike/mindone/assets/8342575/e43ae863-1bfb-49b2-9a61-5b22ffbf4864) and [depth image](https://github.com/zhtmike/mindone/assets/8342575/ac0e410f-5069-42a0-b5a8-45716c9d1254), saved them as `assets/statue.png` and `assets/structure_controls/depth.png` respectively. Then, run the following command in your terminal:
```bash
python controlnet_sdxl.py --img path_of_the_image --control_image path_of_the_control_image --weight path_of_the_ckpt --num_cols 3
```
The `--num_cols` flag denotes the number of output images in a single trial. The output images will be saved under `outputs/demo/SDXL-base-1.0` directory. Here are some examples (The leftmost images is the input image, the second one from left is the control image, while the remaining images are the outputs.):

```bash
python controlnet_sdxl.py --img assets/statue.png --control_img assets/structure_controls/depth.png --ncols 3 --seed 0
```

<p align="center"><img width="1200" src="https://github.com/zhtmike/mindone/assets/8342575/2f0134c1-1e67-42a9-ac8a-8f82d4f6ed66"/>
<br><em>An example of ControlNet. (SD-XL, depth)</em></p>

### Image Variation (SD-1.5)

To run the image variation task on SD-1.5, you can use the `image_variation_sd.py` script. First, prepare the input image and the converted checkpoint. As an example, you may download the [input image](https://github.com/zhtmike/mindone/assets/8342575/dd266d78-2b4b-4159-ad95-5a6afc138235) and saved it as `assets/woman.png`. Then, run the following command in your terminal:
```bash
python image_variation_sd.py --img path_of_the_image --ckpt_path path_of_the_ckpt --n_samples 4
```
The `--n_samples` flag denotes the number of output images in a single trial. The output images will be saved under `outputs/demo/SD/samples` directory. Here are some examples (The image on the left is the original input, while the remaining images are the outputs.):

```bash
python image_variation_sd.py --img assets/woman.png
```
<p align="center"><img width="1200" src="https://github.com/zhtmike/mindone/assets/8342575/a42eca42-3c73-4f0d-865e-66a8347f8a67"/>
<br><em>An example of image variation. (SD-1.5)</em></p>

### Image-To-Image (SD-1.5)

To run the image-to-image task on SD-1.5, you can use the `image_to_image_sd.py` script. First, prepare the input image, reference image and the converted checkpoint. As an example, you may download the [input image](https://github.com/zhtmike/mindone/assets/8342575/1b25fa59-96ff-439c-9d33-75d6bbc96f2d) and [reference image](https://github.com/zhtmike/mindone/assets/8342575/2e1a2c37-4436-484a-948d-865febadeedc), saved them as `assets/river.png` and `assets/vermeer.jpg` respectively. Then, run the following command in your terminal:
```bash
python image_to_image_sd.py --img path_of_the_image --ref_img path_of_the_ref_image --ckpt_path path_of_the_ckpt --n_samples 4 --strength 0.6
```
The `--n_samples` flag denotes the number of output images in a single trial. The `--strength` flag controls the image similarity betweeen the reference image and the output image. The output images will be saved under `outputs/demo/SD/samples` directory. Here are some examples (The leftmost images is the input image, and the second one from left is the reference image, while the remaining images are the outputs.):

```bash
python image_to_image_sd.py --img assets/river.png --ref_img assets/vermeer.jpg
```
<p align="center"><img width="1200" src="https://github.com/zhtmike/mindone/assets/8342575/8eec2237-b339-47b5-adbf-1ec876100623"/>
<br><em>An example of image-to-image. (SD-1.5)</em></p>

### Image Inpainting (SD-1.5)

To run the image inpainting task on SD-1.5, you can use the `image_inpainting_sd.py` script. First, prepare the input image, reference image, the mask of the reference image, and the converted checkpoint. As an example, you may download the [input image](https://github.com/zhtmike/mindone/assets/8342575/e4837c29-40ef-468c-b94e-6df1e3485211), [reference image](https://github.com/zhtmike/mindone/assets/8342575/fdbcfc46-d2ba-4c0a-818a-ad08f931d4a1), and [mask image](https://github.com/zhtmike/mindone/assets/8342575/d2b27d3a-6125-4c37-bf13-7a6b6bde7495) saved them as `assets/girl.png`, `assets/inpainting/image.png` and `assets/inpainting/mask.png` respectively. Then, run the following command in your terminal:
```bash
python image_inpainting_sd.py --img path_of_the_image --ref_img path_of_the_ref_image --ref_mask path_of_the_mask_image --ckpt_path path_of_the_ckpt --n_samples 4 --strength 0.7
```
The `--n_samples` flag denotes the number of output images in a single trial. The `--strength` flag controls the image similarity betweeen the reference image and the output image. The output images will be saved under `outputs/demo/SD/samples` directory. Here are some examples (The leftmost images is the input image, the second one from left is the reference image, and the third one from left is the mask, while the remaining images are the outputs.):

```bash
python image_inpainting_sd.py --img assets/girl.png --ref_img assets/inpainting/image.png --ref_mask assets/inpainting/mask.png --seed 0
```
<p align="center"><img width="1200" src="https://github.com/zhtmike/mindone/assets/8342575/39818911-5162-4b40-a953-ddf9161255cb"/>
<br><em>An example of image inpainting. (SD-1.5)</em></p>

### ControlNet (SD-1.5)

To run the ControlNet task on SD-1.5, you can use the `controlnet_sd.py` script. First, download the SD-1.5 ControlNet checkpoint (in `.safetensors` format) from (https://huggingface.co/lllyasviel), save them under `checkpoints/sd_models`. Prepare the converted checkpoint by running

```bash
python tools/merge_ckpts_sd.py --controlnet path_of_the_controlnet_checkpoint
```

The converted checkpoint `sd_v1.5_ip_adapter.ckpt` will be saved under `checkpoints/sd_models/merged` directory. Then, prepare then input image and the control image, as an example, you may download the [input image 1](https://github.com/zhtmike/mindone/assets/8342575/e43ae863-1bfb-49b2-9a61-5b22ffbf4864), [input image 2](https://github.com/zhtmike/mindone/assets/8342575/e4837c29-40ef-468c-b94e-6df1e3485211), [depth image](https://github.com/zhtmike/mindone/assets/8342575/ac0e410f-5069-42a0-b5a8-45716c9d1254), and [openpose image](https://github.com/zhtmike/mindone/assets/8342575/0791217b-acf2-4c8f-aa41-da2911b8058d), saved them as `assets/statue.png`, `assets/girl`, `assets/structure_controls/depth.png`, `assets/structure_controls/openpose.png` respectively. Then, run the following command in your terminal:
```bash
python controlnet_sd.py --img path_of_the_image --control_image path_of_the_control_image --ckpt_path path_of_the_ckpt --n_samples 4
```
The `--n_samples` flag denotes the number of output images in a single trial. The output images will be saved under `outputs/demo/SD/samples` directory. Here are some examples (The leftmost images is the input image, the second one from left is the control image, while the remaining images are the outputs.):

```bash
python controlnet_sd.py --img assets/statue.png --control_img assets/structure_controls/depth.png
```

<p align="center"><img width="1200" src="https://github.com/zhtmike/mindone/assets/8342575/03590d89-a165-4063-bd4e-ac31458a573b"/>
<br><em>An example of ControlNet. (SD-1.5, depth)</em></p>

```bash
python controlnet_sd.py --img assets/girl.png --control_img assets/structure_controls/openpose.png
```
<p align="center"><img width="1200" src="https://github.com/zhtmike/mindone/assets/8342575/d1582945-dcb4-41d6-91dd-4bdb4e05a82d"/>
<br><em>An example of ControlNet. (SD-1.5, openpose)</em></p>

## Model Training

### Text-image Dataset Preparation

The text-image pair dataset for finetuning should follow the file structure below

```text
dir
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── img_txt.csv
```

`img_txt.csv` is the annotation file in the following format
```text
dir,text
img1.jpg,a cartoon character with a potted plant on his head
img2.jpg,a drawing of a green pokemon with red eyes
img3.jpg,a red and white ball with an angry look on its face
```

For convenience, we have prepared one public text-image dataset obeying the above format.

- [pokemon-blip-caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 833 pokemon-style images with BLIP-generated captions.

To use it, please download `pokemon_blip.zip` from the [openi dataset website](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets). Then unzip them on your local directory, e.g. `./datasets/pokemon_blip`.


### IP-Adapter Finetune (SD-XL)

We will use `train_sdxl.py` script to finetune the IP-Adapter (SD-XL). Run the following command to launch finetuning:

```bash
python train_sdxl.py \
    --data_path ./datasets/pokemon_blip/train \
    --save_ckpt_interval 5000 \
    --weight checkpoints/sdxl_models/merged/sd_xl_base_1.0_ms_ip_adapter.ckpt
```

To improve model performance, it is recommended to run distributed training on at least four devices. The corresponding command is as follows:

```bash
mpirun -n 4 python train_sdxl.py \
    --data_path ./datasets/pokemon_blip/train \
    --is_parallel True \
    --save_ckpt_interval 5000 \
    --weight checkpoints/sdxl_models/merged/sd_xl_base_1.0_ms_ip_adapter.ckpt
```

> Note: to modify other important hyper-parameters, please refer to training config file `configs/training/sd_xl_base_finetune_910b.yaml`.

After training, the checkpoint of the finetuned IP-Adpater will be saved in `runs/YYYY.mm.DD-HH:MM:SS/weights/SDXL-base-1.0-200000_ip_only.ckpt` by default.

Below are some arguments in the config file that you may want to tune for a better performance on your dataset:

- `per_batch_size`: the number of batch size for training.
- `base_learning_rate`: the learning rates for training.
- `total_step`: total number of training steps

For more argument illustration, please run `python train_sdxl.py -h`.

Once you have finished the model finetuning, you need to merge the weight from stable diffusion XL, image encoder, and *finetuned* IP-Adapter into a single file, which can be loaded by the inference script later. To merge the checkpoint, please run the following command:

```bash
python tools/merge_ckpts_sdxl.py \
    --ip_adpater runs/YYYY.mm.DD-HH:MM:SS/weights/SDXL-base-1.0-200000_ip_only.ckpt \
    --out runs/YYYY.mm.DD-HH:MM:SS/weights/SDXL-base-1.0-200000_full.ckpt
```

The full model is saved as `SDXL-base-1.0-200000_full.ckpt`, which can be loaded by `image_variation_sdxl.py` directly. Here are some inference result (The image on the left is the original input, while the remaining images are the outputs.):

<p align="center"><img width="1200" src="https://github.com/zhtmike/mindone/assets/8342575/a336ca54-22f6-4548-b6a2-898265099109"/>
<p align="center"><img width="1200" src="https://github.com/zhtmike/mindone/assets/8342575/f68c94a7-4050-4083-9e3e-7d8ef00f4e82"/>
<br><em>Finetuned result on Pokemon dataset. (SD-XL)</em></p>

### IP-Adapter Finetune (SD-1.5)

We will use `train_sd.py` script to finetune the IP-Adapter (SD-1.5). Run the following command to launch finetuning:

```bash
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
python train_sd.py \
    --data_path ./datasets/pokemon_blip/train \
    --pretrained_model_path checkpoints/sd_models/merged/sd_v1.5_ip_adapter.ckpt
```

> Note: to modify other important hyper-parameters, please refer to training config file `configs/training/sd_v15_finetune_910b.yaml`.

After training, the checkpoint of the finetuned IP-Adpater will be saved in `runs/ckpt/sd-2000.ckpt` by default.

Below are some arguments in the config file that you may want to tune for a better performance on your dataset:

- `train_batch_size`: the number of batch size for training.
- `start_learning_rate`: the learning rates for training.
- `epochs`: total number of training epochs

For more argument illustration, please run `python train_sd.py -h`.

Once you have finished the model finetuning, you need to merge the weight from stable diffusion 1.5, image encoder, and *finetuned* IP-Adapter into a single file, which can be loaded by the inference script later. To merge the checkpoint, please run the following command:

```bash
python tools/merge_ckpts_sd.py \
    --ip_adpater runs/ckpt/sd-2000.ckpt \
    --out runs/ckpt/sd-2000_full.ckpt
```

The full model is saved as `sd-2000_full.ckpt`, which can be loaded by `image_variation_sd.py` directly. Here are some inference result (The image on the left is the original input, while the remaining images are the outputs.):

<p align="center"><img width="1200" src="https://github.com/zhtmike/mindone/assets/8342575/d20177bc-2270-4bc6-b127-91946f0428fc"/>
<p align="center"><img width="1200" src="https://github.com/zhtmike/mindone/assets/8342575/b5095eab-4c19-4015-9436-9ad9046ab90a"/>
<br><em>Finetuned result on Pokemon dataset. (SD-1.5)</em></p>

### IP-Adapter Training From scratch

If you want to train the IP-Adapter from scratch, follow these steps:

1. Prepare the checkpoint without IP-Adapter weight by running the following command:

```bash
python tools/merge_ckpts_sdxl.py --skip_ip --out checkpoints/sdxl_models/merged/sd_xl_base_1.0_ms_ip_adapter_train.ckpt
```

2. Prepare a relatively large amount of training data for model training, such as LAION-2B and COYO-700M. For illustration purposes, we will use COCO2017 as training data. Download COCO2017 training image, val images and train/val annotations from [https://cocodataset.org/]

3. Convert the COCO label from JSON format into text format by running the following command:

```bash
python tools/prepare_coco.py --label path_of_captions_train2017.json --image path_of_train2017
```

4. Put the converted text label file `img_txt.csv` under the COCO directory `train2017`,

5. Start the training by running the following command:

    ```bash
    python train_sdxl.py \
        --data_path COCO2017/train2017 \
        --weight checkpoints/sd_xl_base_1.0_ms_ip_adapter_train.ckpt
    ```

    If you want to run the training on 8 devices, you can use the following command:

    ```bash
    mpirun -n 8 python train_sdxl.py \
        --data_path COCO2017/train2017 \
        --is_parallel True \
        --weight checkpoints/sd_xl_base_1.0_ms_ip_adapter_train.ckpt
    ```

    You may also need to increase the training steps and adjust the learning rate in the configuration file to improve the model’s performance.

## Acknowledgments

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, Wei Yang. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. arXiv:2308.06721, 2023.
