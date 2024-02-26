# ControlNet based on Stable Diffusion XL
> [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)

## Introduction
ControlNet controls pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small. Large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like canny edge maps, segmentation maps, keypoints, etc.

<p align="center">
   <img src="https://github.com/Gaohan123/mindone/assets/20148503/c5c27f00-3c20-479c-a540-70a0c8db0d48" width=700 />
</p>
<p align="center">
  <em> Fig 1. Illustration of a ControlNet [<a href="#reference">1</a>] </em>
</p>


## Dependency

- AI framework: MindSpore >= 2.2
- Hardware: Ascend 910*

```shell
cd examples/stable_diffusion_xl
pip install -r requirement.txt
```

## Inferece

### Prepare model weight

**1. Convert trained weight from Diffusers**

  **Step1**: Convert SDXL-base-1.0 model weight from Diffusers to MindONE, refer to [here](../../GETTING_STARTED.md#convert-pretrained-checkpoint). Get `sd_xl_base_1.0_ms.ckpt`.

  **Step2**: Since ControlNet acts like a plug-in to the SDXL, we convert the ControlNet weight `diffusion_pytorch_model.safetensors` from [diffusers/controlnet-canny-sdxl-1.0](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/tree/main)
  to MindSpore version and then merge it into the SDXL-base-1.0 MindONE model weight (`sd_xl_base_1.0_ms.ckpt`). Eventually, we get the ControlNet + SDXL-base-1.0 MindONE model weight (`sd_xl_base_1.0_controlnet_canny_ms.ckpt`).


  ```shell
  cd tools/controlnet_conversion

  python convert_weight.py  \
      --weight_torch_controlnet /PATH TO/diffusion_pytorch_model.safetensors  \
      --weight_ms_sdxl /PATH TO/sd_xl_base_1.0_ms.ckpt  \
      --output_ms_ckpt_path /PATH TO/sd_xl_base_1.0_controlnet_canny_ms.ckpt
  ```

  > Note: The ControlNet weight parameters name mapping between Diffusers and MindONE is prepared: `tools/controlnet_conversion/controlnet_ms2torch_mapping.yaml`.

**2. Or train your ControlNet using MindONE, check [Training](#training) section below**

### Prepare control signals

Stable Diffusion XL with ControlNet can generate images following the input control signal (e.g. canny edge). You can either prepare (1) a raw image (Fig 2) to be extracted control signal from, or (2) the control signal image itself (Fig 3).

<div align="center">
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/f8a7ef86-3d4a-4d07-b99e-46156c356e73" width=30% />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/7eaff4e2-d9a4-44e6-a059-e8e1074f2301" width=30% />
</div>
<p align="center">
<em> Fig 2. raw image to be extracted control signal </em>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<em> Fig 3. control signal image (canny edge) </em>
</p>


### Generate images

Please refer to [`scripts/run_infer_base_controlnet.sh`](scripts/run_infer_base_controlnet.sh).

```shell
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base_controlnet.yaml \
  --weight checkpoints/sd_xl_base_1.0_controlnet_canny_ms.ckpt \
  --guidance_scale 9.0 \
  --controlnet_mode canny \
  --control_image_path /PATH TO/dog2.png \
  --prompt "cute dog, best quality, extremely detailed"   \
```

Key arguments:
- `weight`: path to the model weight, refer to [Prepare model weight](#prepare-model-weight) section.
- `guidance_scale`: the guidance scale for txt2img and img2img tasks. For NoDynamicThresholding, uncond + guidance_scale * (uncond - cond). **Note that this scale could heavily impact the inference result.**
- `controlnet_mode`: Control mode for controlnet, supported mode: 'raw': use the image itself as control signal; 'canny': use canny edge detector to extract control signal from input image.
- `control_image_path`: path of input image for controlnet.
- `prompt`: positve text prompt for image generation.

You can check all arguments description by running `python demo/sampling_without_streamlit.py -h`.


### Inference results

<div align="center">
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/f8a7ef86-3d4a-4d07-b99e-46156c356e73" width=30% />
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/7eaff4e2-d9a4-44e6-a059-e8e1074f2301" width=30% />
<img src="https://github.com/zhtmike/mindone/assets/20376974/9501f7e9-5bd0-45ac-8662-6866d27fc645" width=30% />
</div>
<p align="center">
<em> Fig 4. From left to right: raw image - extracted canny edge - inference result. </em>
</br>
<em> Prompt: "cute dog, best quality, extremely detailed". </em>
</p>

<div align="center">
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/ecdc18c9-36bf-4d49-b7b2-6216400f1d5a" width=30% />
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/00f0ed2c-c078-4ed3-bee6-cb8c66fb36fd" width=30% />
<img src="https://github.com/zhtmike/mindone/assets/20376974/28cf7e42-3f4c-4ee8-b4e0-e9a0f8244c43" width=30% />
</div>
<p align="center">
<em> Fig 5. From left to right: raw image - extracted canny edge - inference result. </em>
</br>
<em> Prompt: "beautiful bird standing on a trunk, natural color, best quality, extremely detailed". </em>
</p>

## Training

### Prepare init model weight
**Step1**: Convert SDXL-base-1.0 model weight from Diffusers to MindONE, refer to [here](../../GETTING_STARTED.md#convert-pretrained-checkpoint). Get `sd_xl_base_1.0_ms.ckpt`.

**Step2**:

```shell
cd tools/controlnet_conversion
python init_weight.py
```

The parameters of `zero_conv`, `input_hint_block` and `middle_block_out` blocks are randomly initialized in ControlNet. Other parameters of ControlNet are copied from SDXL pretrained weight `sd_xl_base_1.0_ms.ckpt` (referring to [here](GETTING_STARTED.md#convert-pretrained-checkpoint)).

### Prepare dataset

We use [Fill50k dataset](https://huggingface.co/datasets/HighCWu/fill50k) to train the model to generate images following the edge control. The directory struture of Fill50k dataset is shown below.

```text
DATA_PATH
  ├── prompt.json
  ├── source
  │   ├── 0.png
  │   ├── 1.png
  │   └── ...
  └── target
      ├── 0.png
      ├── 1.png
      └── ...
```

Images in `target/` are raw images. Images in `source/` are the canny edge/segementation/other control images extracted from the corresponding raw images. For example, `source/img0.png` is the canny edge image of `target/img0.png`.

`prompt.json` is the annotation file with the following format.

```json
{"source": "source/0.png", "target": "target/0.png", "prompt": "pale golden rod circle with old lace background"}
{"source": "source/1.png", "target": "target/1.png", "prompt": "light coral circle with white background"}
{"source": "source/2.png", "target": "target/2.png", "prompt": "aqua circle with light pink background"}
{"source": "source/3.png", "target": "target/3.png", "prompt": "cornflower blue circle with light golden rod yellow background"}
```

> Note: if you want to use your own dataset for training, please follow the directory and file structure shown above.

### Launch training

Please refer to [`scripts/run_train_base_controlnet.sh`](scripts/run_train_base_controlnet.sh).

```shell
nohup mpirun -n 8 --allow-run-as-root python train_controlnet.py \
    --data_path DATA_PATH \
    --weight PATH TO/sd_xl_base_1.0_ms_controlnet_init.ckpt \
    --config configs/training/sd_xl_base_finetune_controlnet_910b.yaml \
    --total_step 300000 \
    --per_batch_size 2 \
    --group_lr_scaler 10.0 \
    --save_ckpt_interval 10000 \
    --max_num_ckpt 5 \
    > train.log 2>&1 &
```

⚠️ Some key points about ControlNet + SDXL training:
- The parameters of `zero_conv`, `input_hint_block` and `middle_block_out` blocks are randomly initialized in ControlNet, which are very hard to train. We scale up (x10 by default) the base learning rate for training parameters specifically. You can set the scale value by `args.group_lr_scaler`.
- As mentioned in ControlNet paper[1] and [repo](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md#more-consideration-sudden-converge-phenomenon-and-gradient-accumulation), there is a sudden convergence phenomenon in ControlNet training, which means the training steps should be large enough to let the training converge SUDDENLY and then generate images following the control signals. For ControlNet + SDXL, the training steps should be even much more larger.
- As mentioned in ControlNet paper[1], randomly dropping 50% text prompt during training is very helpful for ControlNet to learn the control signals. Don't miss that.

### Training results

Key settings:

|base_learning_rate |group_lr_scaler | global batch size (#NPUs * bs per NPU)| total_step | inference guidance_scale |
|-----------|----------------|--------------|----------------|------------|
|  4.0e-5   |    10.0        |  64 (32 x 2) |   220k         |  15.0      |

<br>
Ground truth:

<div align="center">
<kbd>
<img src="https://github.com/zhtmike/mindone/assets/20376974/db6e42cf-63db-44f6-afeb-691f65ab991d" width=24% />
<img src="https://github.com/zhtmike/mindone/assets/20376974/dc34013e-d7a5-4d6f-a54b-28310e8cc252" width=24% />
<img src="https://github.com/zhtmike/mindone/assets/20376974/b6a68ace-641c-4bef-96ab-8f8ad09babd5" width=24% />
<img src="https://github.com/zhtmike/mindone/assets/20376974/122095f1-338b-4d73-bc48-065c9f41e070" width=24% />
</kbd>
</div>


<br>
Our prediction:

<div align="center">
<kbd>
<img src="https://github.com/zhtmike/mindone/assets/20376974/580eca17-160f-430c-952e-9b1373acf952" width=24% />
<img src="https://github.com/zhtmike/mindone/assets/20376974/d612328d-4520-4ada-8c55-76d3a7b164bc" width=24% />
<img src="https://github.com/zhtmike/mindone/assets/20376974/1f685df0-7095-40cd-b353-7655073977c3" width=24% />
<img src="https://github.com/zhtmike/mindone/assets/20376974/442238fc-2c5d-4d26-a3db-ea3dcbc7d72b" width=24% />
</kbd>
</div>

<br>
Prompts (correspond to the images above from left to right):

```
"light coral circle with white background"
"light sea green circle with dark salmon background"
"medium sea green circle with black background"
"dark turquoise circle with medium spring green background"
```

## Reference
[1] [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)
