# T2I-Adapter

- [介绍](#introduction)
- [预训练模型](#pretrained-models)
- [推理和示例](#inference-and-examples)
    - [单独适配器](#individual-adapters)
    - [组合适配器](#combined-adapters)
- [训练](#training)
- [致谢](#acknowledgements)

## 介绍

T2I-适配器是一种简单且轻量级的网络，为稳定扩散模型提供额外的视觉指导，除了内置的文本指导外，还利用了隐式学习的能力。这些适配器充当了SD模型的插件，使其易于集成和使用。T2I-适配器的总体架构如下：

<p align="center"><img width="700" alt="T2I-Adapter Architecture" src="https://github.com/mindspore-lab/mindone/assets/16683750/a957957b-1576-458a-9441-436c82fb9320"/>
<br><em>整体T2I-适配器架构</em></p>

这种架构有多个优点：

- T2I-适配器**不会**影响稳定扩散模型的权重。此外，训练T2I-适配器**不需要**训练SD模型本身。
- **简单且轻量级**：完整适配器有77M参数，轻量级适配器有5M参数。
- **可组合**：可以组合多个适配器以实现多条件控制。
- **可泛化**：只要它们是从相同的模型微调而来的（例如，使用在SD 1.4上训练的T2I-适配器与SD 1.5或Anything anime模型），就可以直接用于自定义模型。


## 预训练模型

<div align="center">

| SD Compatibility | Task          | SD Train Version | Dataset                                | Recipe                             | Weights                                                                                                          |
|:----------------:|---------------|:----------------:|----------------------------------------|------------------------------------|------------------------------------------------------------------------------------------------------------------|
|       SDXL       | Canny         |     SDXL 1.0     | LAION-Aesthetics V2 (3M)               |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/adapter_xl_canny-aecfc7d6.ckpt)           |
|                  | Depth (MiDaS) |     SDXL 1.0     | LAION-Aesthetics V2 (3M)               |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/adapter_xl_depth-5ce5acf2.ckpt)           |
|                  | LineArt       |     SDXL 1.0     | LAION-Aesthetics V2 (3M)               |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/adapter_xl_lineart-6110edd0.ckpt)         |
|                  | OpenPose      |     SDXL 1.0     | LAION-Aesthetics V2 (3M)               |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/adapter_xl_openpose-88397cd1.ckpt)        |
|                  | Sketch        |     SDXL 1.0     | LAION-Aesthetics V2 (3M)               |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/adapter_xl_sketch-98dbd348.ckpt)          |
|                  |               |                  |                                        |                                    |                                                                                                                  |
|       2.x        | Segmentation  |       2.1        | [COCO-Stuff](#segmentation-coco-stuff) | [yaml](configs/sd_v2.1_train.yaml) | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_seg_sd21-86d4e0db.ckpt)        |
|                  |               |                  |                                        |                                    |                                                                                                                  |
|       1.x        | Canny         |       1.5        | [COCO-Stuff](#segmentation-coco-stuff) |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_canny_sd15v2-c484cd69.ckpt)    |
|                  | Color         |       1.4        | LAION-Aesthetics V2 (625K)             |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_color_sd14v1-7cb31ebd.ckpt)    |
|                  | Depth (MiDaS) |       1.5        | LAION-Aesthetics V2 (625K)             |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_depth_sd15v2-dc86209b.ckpt)    |
|                  | KeyPose       |       1.4        | LAION-Aesthetics V2 (625K)             |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_keypose_sd14v1-ee27ccf0.ckpt)  |
|                  | OpenPose      |       1.4        | LAION-Aesthetics V2 (625K)             |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_openpose_sd14v1-ebcdb5cb.ckpt) |
|                  | Segmentation  |       1.4        | [COCO-Stuff](#segmentation-coco-stuff) |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_seg_sd14v1-1d2e8478.ckpt)      |
|                  | Sketch        |       1.5        | [COCO-Stuff](#segmentation-coco-stuff) |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_sketch_sd15v2-6c537e26.ckpt)   |
|                  | Style         |       1.4        |                                        |                                    | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_style_sd14v1-a620ae97.ckpt)    |

</div>

**注意**：

- 如在 [介绍](#Introduction) 中提到的那样，T2I 适配器具有良好的泛化性，因此可以与自定义模型一起使用（只要它们是从相同的模型微调而来的），例如，可以使用在 SD 1.4 上训练的 T2I 适配器与 SD 1.5 或 Anything anime 模型一起使用。<br>
- :warning: SD 1.x 上训练的 T2I 适配器与 SD 2.x 不兼容，因为架构上存在差异。

上述权重是从 PyTorch 版本转换而来的。如果您想转换另一个自定义模型，可以使用 `t2i_tools/convert.py`。例如：


```shell
python t2i_tools/convert.py --diffusion_model SDXL \
--pt_weights_file PATH_TO_YOUR_TORCH_MODEL \
--task CONDITION \
--out_dir PATH_TO_OUTPUT_DIR
```

## 推理和示例

有关可能的参数和用法的详细信息，请执行以下命令：

```shell
python adapter_image2image_sd.py --help # for SD
python adapter_image2image_sdxl.py --help # for SDXL
```

另外，您可以在下面找到一些SD和SDXL的示例用例。
示例中使用的条件图像可以在[此处](https://huggingface.co/TencentARC/T2I-Adapter/tree/main/examples)
和[此处](https://huggingface.co/TencentARC/t2i-adapter-canny-sdxl-1.0)找到

### 单个适配器

#### Canny 适配器

##### SD


<p align="center">
<img width="256" alt="SD Canny input" src="https://github.com/mindspore-lab/mindone/assets/16683750/c6ae6ca5-356e-4028-9dd9-930e1be8adf2"/>
<img width="256" alt="SD Canny output" src="https://github.com/mindspore-lab/mindone/assets/16683750/1b63916e-417d-447a-ad3f-ce362db19f35"/>
<br><em>Prompt: Cute toy, best quality, extremely detailed</em>
</p>

<details>
<summary>执行命令</summary>

```shell
python adapter_image2image_sd.py \
--version 1.5 \
--prompt "Cute toy, best quality, extremely detailed" \
--adapter_ckpt_path models/t2iadapter_canny_sd15v2-c484cd69.ckpt \
--ddim \
--adapter_condition canny \
--condition_image samples/canny/toy_canny.png
```

</details>

##### SDXL

<p align="center">
<img width="256" alt="SDXL Canny input" src="https://github.com/mindspore-lab/mindone/assets/16683750/b91de7f5-d498-4945-9cbd-d558f4e4858c"/>
<img width="256" alt="SDXL Canny output" src="https://github.com/mindspore-lab/mindone/assets/16683750/2b373694-460f-476b-b6c8-9ffd6b1ac9df"/>
<br><em>Prompt: Mystical fairy in real, magic, 4k picture, high quality</em>
</p>

<details>
<summary>执行命令</summary>

```shell
python adapter_image2image_sdxl.py \
--config=configs/sdxl_inference.yaml \
--SDXL.checkpoints=models/sd_xl_base_1.0_ms.ckpt \
--adapter.condition=canny \
--adapter.ckpt_path=models/adapter_xl_canny-aecfc7d6.ckpt \
--adapter.cond_weight=0.8 \
--adapter.image=samples/canny/figs_SDXLV1.0_cond_canny.png \
--prompt="Mystical fairy in real, magic, 4k picture, high quality" \
--negative_prompt="extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured" \
--n_samples=4
```

</details>

#### LineArt Adapter

##### SDXL

<p align="center">
<img width="480" alt="SDXL LineArt input" src="https://github.com/mindspore-lab/mindone/assets/16683750/fed6d28a-96c2-4463-8c91-1ff0252be1d0"/>
<img width="480" alt="SDXL LineArt output" src="https://github.com/mindspore-lab/mindone/assets/16683750/2c77f8c6-52c3-4e89-a3e8-437da2c735d6"/>
<br><em>Prompt: Ice dragon roar, 4k photo</em>
</p>

<details>
<summary>执行命令</summary>

```shell
python adapter_image2image_sdxl.py \
--config=configs/sdxl_inference.yaml \
--SDXL.checkpoints=models/sd_xl_base_1.0_ms.ckpt \
--adapter.condition=lineart \
--adapter.ckpt_path=models/adapter_xl_lineart-6110edd0.ckpt \
--adapter.cond_weight=0.8 \
--adapter.image=samples/lineart/figs_SDXLV1.0_cond_lin.png \
--prompt="Ice dragon roar, 4k photo" \
--negative_prompt="anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured" \
--n_samples=4
```

</details>

#### Spatial Palette (Color) Adapter (SD only)

<p align="center">
<img width="320" alt="Color input" src="https://github.com/mindspore-lab/mindone/assets/16683750/01e41f01-188e-4331-adac-597d2e3fc9f7"/>
<img width="320" alt="Color output" src="https://github.com/mindspore-lab/mindone/assets/16683750/3e1f51ff-1acf-4b80-bac4-aeabf95c564f"/>
<br><em>Prompt: A photo of scenery</em>
</p>

<details>
<summary>执行命令</summary>

```shell
python adapter_image2image_sd.py \
--version 1.5 \
--prompt "A photo of scenery" \
--adapter_ckpt_path models/t2iadapter_color_sd14v1-7cb31ebd.ckpt \
--ddim \
--adapter_condition color \
--condition_image samples/color/color_0002.png \
--scale 9
```

</details>

#### Depth Adapter

##### SD

<p align="center">
<img height="366" alt="Depth input" src="https://github.com/mindspore-lab/mindone/assets/16683750/592ee66b-d3cc-428b-8efc-f3f89a02e5bd"/>
<img height="366" alt="Depth output" src="https://github.com/mindspore-lab/mindone/assets/16683750/30efde92-eaaa-4e74-8c4f-c7fa8535bfc3"/>
<br><em>Prompt: desk, best quality, extremely detailed</em>
</p>

<details>
<summary>执行命令</summary>

```shell
python adapter_image2image_sd.py \
--version 1.5 \
--prompt "desk, best quality, extremely detailed" \
--adapter_ckpt_path models/t2iadapter_depth_sd15v2-dc86209b.ckpt \
--ddim \
--adapter_condition depth \
--condition_image samples/depth/desk_depth.png
```

</details>

#### OpenPose Adapter

##### SD

<p align="center">
<img width="256" alt="OpenPose input" src="https://github.com/mindspore-lab/mindone/assets/16683750/04b8aa78-3914-46c0-bf8c-37acc064cd4d"/>
<img width="256" alt="OpenPose output" src="https://github.com/mindspore-lab/mindone/assets/16683750/56db6db4-aa77-4b15-ae03-485bb8144e71"/>
<br><em>Prompt: Iron man, high-quality, high-res</em>
</p>

<details>
<summary>执行命令</summary>

```shell
python adapter_image2image_sd.py \
--version 1.5 \
--prompt "Iron man, high-quality, high-res" \
--adapter_ckpt_path models/t2iadapter_openpose_sd14v1-ebcdb5cb.ckpt \
--ddim \
--adapter_condition openpose \
--condition_image samples/openpose/iron_man_pose.png
```

</details>

#### Segmentation Adapter

##### SD

<p align="center">
<img width="320" alt="Segmentation input" src="https://github.com/mindspore-lab/mindone/assets/16683750/f7486cf2-1c0d-4b4f-bb04-fcfc1aeeae87"/>
<img width="320" alt="Segmentation output SDv1.5" src="https://github.com/mindspore-lab/mindone/assets/16683750/6e3b547d-1c42-4b4b-9751-26bd9d80aee4"/>
<img width="320" alt="Segmentation output SDv2.1" src="https://github.com/mindspore-lab/mindone/assets/16683750/31a95dac-00cd-47a3-85b6-e3732ff96ec6"/>
<br><em>Prompt: A black Honda motorcycle parked in front of a garage, best quality, extremely detailed</em>
<br><em>SD1.5 output on the left and SD2.1 output on the right.</em>
</p>

<details>
<summary>执行命令</summary>

```shell
# StableDiffusion v2.1
python adapter_image2image_sd.py \
--version 2.1 \
--prompt "A black Honda motorcycle parked in front of a garage, best quality, extremely detailed" \
--ckpt_path=models/sd_v2-1_base-7c8d09ce.ckpt \
--adapter_ckpt_path models/t2iadapter_seg_sd21-86d4e0db.ckptt \
--ddim \
--adapter_condition seg \
--condition_image samples/seg/motor.png
```

```shell
# StableDiffusion v1.5
python adapter_image2image_sd.py \
--version 1.5 \
--prompt "A black Honda motorcycle parked in front of a garage, best quality, extremely detailed" \
--adapter_ckpt_path models/t2iadapter_seg_sd14v1-1d2e8478.ckpt \
--ddim \
--adapter_condition seg \
--condition_image samples/seg/motor.png
```

</details>

#### Sketch Adapter

##### SD

<p align="center">
<img width="320" alt="SD Sketch input" src="https://github.com/mindspore-lab/mindone/assets/16683750/feef1c90-6ed9-4af4-a3ee-a949000fcc59"/>
<img width="320" alt="SD Sketch output" src="https://github.com/mindspore-lab/mindone/assets/16683750/e5661fe2-dc29-4f1b-a901-387329b06e52"/>
<br><em>Prompt: A car with flying wings</em>
</p>

<details>
<summary>执行命令</summary>

```shell
python adapter_image2image_sd.py \
--version 1.5 \
--prompt "A car with flying wings" \
--adapter_ckpt_path models/t2iadapter_sketch_sd15v2-6c537e26.ckpt \
--ddim \
--adapter_condition sketch \
--condition_image samples/sketch/car.png \
--cond_tau 0.5
```

</details>

##### SDXL

<p align="center">
<img width="256" alt="SDXL Sketch input" src="https://github.com/mindspore-lab/mindone/assets/16683750/81f4098b-9ba3-424b-a335-f9d5e96988d5"/>
<img width="256" alt="SDXL Sketch output" src="https://github.com/mindspore-lab/mindone/assets/16683750/98c6adf7-d0f3-4efc-b6d3-1e7b66ced5f1"/>
<br><em>Prompt: a robot, mount fuji in the background, 4k photo, highly detailed</em>
</p>

<details>
<summary>执行命令</summary>

```shell
python adapter_image2image_sdxl.py \
--config=configs/sdxl_inference.yaml \
--SDXL.checkpoints=models/sd_xl_base_1.0_ms.ckpt \
--adapter.condition=sketch \
--adapter.ckpt_path=models/adapter_xl_sketch-98dbd348.ckpt \
--adapter.cond_weight=0.9 \
--adapter.image=samples/sketch/figs_SDXLV1.0_cond_sketch.png \
--prompt="a robot, mount fuji in the background, 4k photo, highly detailed" \
--negative_prompt="extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured" \
--n_samples=4
```

</details>

### 组合适配器

可以将单个T2I适配器组合起来，无需重新训练即可对多个图像进行条件控制。

#### 彩色 + 素描

##### SD

<p align="center">
<img height="256" alt="Sketch input" src="https://github.com/mindspore-lab/mindone/assets/16683750/a30661ce-26e3-42a3-b935-6caabe406d02"/>
<br><em>Prompt: A car with flying wings</em>
</p>

<details>
<summary>执行命令</summary>

```shell
python adapter_image2image_sd.py \
--version 1.5 \
--prompt "A car with flying wings" \
--adapter_ckpt_path models/t2iadapter_sketch_sd15v2.ckpt models/t2iadapter_color_sd14v1.ckpt \
--adapter_condition sketch color \
--condition_image samples/sketch/car.png samples/color/color_0004.png \
--cond_weight 1.0 1.2 \
--ddim
```

</details>

## 训练

下表总结了T2I适配器的训练细节：

<div align="center">

| 任务         | SD 版本 | 数据集                                      | 上下文           | 训练时间         | 吞吐量        | 配方                              |
|--------------|:------:|----------------------------------------------|----------------|----------------|--------------|----------------------------------|
| 分割         |   2.1  | [COCO-Stuff Train](#segmentation-coco-stuff) | D910Ax4-MS2.1-G | 每轮10小时35分钟 | 39.2 img / 秒 | [yaml](configs/sd_v2.1_train.yaml) |

</div>

> 上下文：训练上下文以 {device}x{pieces}-{MS version}{MS mode} 表示，其中 mindspore 模式可以是 G - 图模式 或 F - 使用 ms 函数的 Pynative 模式。例如，D910x8-G 表示使用图模式在 8 片 Ascend 910 NPU 上训练。

### 数据准备

条件图像必须是 RGB 格式。因此，一些数据集在训练之前可能需要预处理。

#### 分割（COCO-Stuff）

分割掩码通常是灰度图像，其值对应于类别标签。但是，为了训练 T2I 适配器，必须将掩码转换为 RGB 图像。要执行此操作，首先下载灰度掩码（[stuffthingmaps_trainval2017.zip](https://github.com/nightrome/cocostuff#downloads)），然后解压缩它们。然后执行以下命令将掩码转换为 RGB 图像：


```shell
python t2i_tools/cocostuff_colorize_mask.py PATH_TO_GRAY_MASKS_DIR PATH_TO_OUTPUT_DIR
```
Annotation labels for COCO-Stuff can be found in [annotations_trainval2017.zip/annotations/](https://github.com/nightrome/cocostuff#downloads).

### Segmentation

After the data preparation is completed, the following command can be used to train T2I-Adapter:


```shell
mpirun --allow-run-as-root -n 4 python train_t2i_adapter_sd.py \
--config configs/sd_v2.1_train.yaml \
--train.dataset.init_args.image_dir PATH_TO_IMAGES_DIR \
--train.dataset.init_args.masks_path PATH_TO_RGB_MASKS_DIR \
--train.dataset.init_args.label_path PATH_TO_LABELS \
--train.output_dir: PATH_TO_OUTPUT_DIR
```

## 评估

T2I-适配器在 COCO-Stuff 验证数据集上进行评估（有关更多详细信息，请参见[数据准备](#segmentation-coco-stuff)）。仅使用每个图像的[第一个提示](https://github.com/TencentARC/T2I-Adapter/issues/65#issuecomment-1541324103)。
以下表格总结了 T2I-适配器的性能：

<div align="center">

| Task         | SD Version | Dataset                                    | FID ↓ | CLIP Score ↑ | Recipe                             |
|--------------|:----------:|--------------------------------------------|-------|--------------|------------------------------------|
| Segmentation |    2.1     | [COCO-Stuff Val](#segmentation-coco-stuff) | 26.10 | 26.32        | [yaml](configs/sd_v2.1_train.yaml) |

</div>

要自行评估 T2I-适配器，首先需要使用 `adapter_image2image.py` 生成图像（有关更多详细信息，请参见[推理和示例](#inference-and-examples)）。然后，要计算 FID，请运行以下命令：

```shell
python examples/stable_diffusion_v2/tools/eval/eval_fid.py \
--backend=ms \
--real_dir=PATH_TO_VALIDATION_IMAGES \
--gen_dir=PATH_TO_GENERATED_IMAGES \
--batch_size=50
```

通过使用 `clip_vit_l_14` 模型来计算 CLIP 分数（更多信息和权重可以在[此处](../stable_diffusion_v2/tools/eval/README.md#clip-score)找到）。要计算分数，请运行以下命令：

```shell
python examples/stable_diffusion_v2/tools/eval/eval_clip_score.py \
--backend=ms \
--config=examples/stable_diffusion_v2/tools/_common/clip/configs/clip_vit_l_14.yaml \
--ckpt_path=PATH/TO/clip_vit_l_14.ckpt \
--tokenizer_path=examples/stable_diffusion_v2/ldm/models/clip/bpe_simple_vocab_16e6.txt.gz \
--image_path_or_dir=PATH_TO_GENERATED_IMAGES \
--prompt_or_path=PATH_TO_PROMPTS \
--save_result=False \
--quiet
```

## 致谢

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, Ying Shan, Xiaohu Qie. T2I-Adapter: Learning
Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models. arXiv:2302.08453, 2023.
