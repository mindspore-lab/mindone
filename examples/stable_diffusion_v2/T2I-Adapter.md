# T2I-Adapter

- [Introduction](#introduction)
- [Pretrained Models](#pretrained-models)
- [Inference and Examples](#inference-and-examples)
    - [Individual Adapters](#individual-adapters)
    - [Combined Adapters](#combined-adapters)
- [Training](#training)
- [Acknowledgements](#acknowledgements)

## Introduction

T2I-Adapters are simple and lightweight networks that provide additional visual guidance to Stable Diffusion models,
in addition to the built-in text guidance, to leverage implicitly learned capabilities. These adapters act as plug-ins
to SD models, making them easy to integrate and use. The overall architecture of T2I-Adapters is as follows:

<p align="center"><img width="700" alt="T2I-Adapter Architecture" src="https://github.com/mindspore-lab/mindone/assets/16683750/a957957b-1576-458a-9441-436c82fb9320"/>
<br><em>Overall T2I-Adapter architecture</em></p>

There are multiple advantages of this architecture:

- T2I-Adapters **do not** affect weights of Stable Diffusion models. Moreover, training T2I-Adapters **does not** requre
  training of an SD model itself.
- **Simple and lightweight**: 77M parameters for full and 5M parameters for light adapters.
- **Composable**: Several adapters can be combined to achieve multi-condition control.
- **Generalizable**: Can be directly used on custom models as long as they fine-tuned from the same model (e.g., use
  T2I-Adapters trained on SD 1.4 with SD 1.5 or Anything anime model).

## Pretrained Models

<div align="center">

| Tasks        | SD version | Weights                                                                                                          |
|--------------|------------|------------------------------------------------------------------------------------------------------------------|
| Canny        | 1.5        | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_canny_sd15v2-c484cd69.ckpt)    |
| Color        | 1.4        | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_color_sd14v1-7cb31ebd.ckpt)    |
| Depth        | 1.5        | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_depth_sd15v2-dc86209b.ckpt)    |
| KeyPose      | 1.4        | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_keypose_sd14v1-ee27ccf0.ckpt)  |
| OpenPose     | 1.4        | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_openpose_sd14v1-ebcdb5cb.ckpt) |
| Segmentation | 1.4        | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_seg_sd14v1-1d2e8478.ckpt)      |
| Sketch       | 1.5        | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_sketch_sd15v2-6c537e26.ckpt)   |
| Style        | 1.4        | [Download](https://download.mindspore.cn/toolkits/mindone/t2i-adapters/t2iadapter_style_sd14v1-a620ae97.ckpt)    |

</div>

> Note: As mentioned in the [Introduction](#Introduction), T2I-Adapters generalize well and thus can be used with custom
> models (as long as they fine-tuned from the same model), e.g., use T2I-Adapters trained on SD 1.4 with SD 1.5 or
> Anything anime model.<br>
> :warning: T2I-Adapters trained on SD 1.x are not compatible with SD 2.x due to difference in the architecture.

The weights above were converted from PyTorch version. If you want to convert another custom model, you can do so by
using `mindone/tools/t2i_adapter_torch_to_ms/convert.py`. For example:

```shell
python mindone/tools/t2i_adapter_torch_to_ms/convert.py PATH_TO_YOUR_TORCH_MODEL
```

## Inference and Examples

The condition images used in the examples can be found
[here](https://huggingface.co/TencentARC/T2I-Adapter/tree/main/examples).

### Individual Adapters

#### Canny Adapter

<p align="center">
<img width="256" alt="Canny input" src="https://github.com/mindspore-lab/mindone/assets/16683750/c6ae6ca5-356e-4028-9dd9-930e1be8adf2"/>
<img width="256" alt="Canny output" src="https://github.com/mindspore-lab/mindone/assets/16683750/1b63916e-417d-447a-ad3f-ce362db19f35"/>
<br><em>Prompt: Cute toy, best quality, extremely detailed</em>
</p>

```shell
python examples/stable_diffusion_v2/adapter_image2image.py \
--version 1.5 \
--prompt "Cute toy, best quality, extremely detailed" \
--adapter_ckpt_path models/t2iadapter_canny_sd15v2-c484cd69.ckpt \
--ddim \
--adapter_condition canny \
--condition_image samples/canny/toy_canny.png
```

#### Spatial Palette (Color) Adapter

<p align="center">
<img width="320" alt="Color input" src="https://github.com/mindspore-lab/mindone/assets/16683750/01e41f01-188e-4331-adac-597d2e3fc9f7"/>
<img width="320" alt="Color output" src="https://github.com/mindspore-lab/mindone/assets/16683750/3e1f51ff-1acf-4b80-bac4-aeabf95c564f"/>
<br><em>Prompt: A photo of scenery</em>
</p>

```shell
python examples/stable_diffusion_v2/adapter_image2image.py \
--version 1.5 \
--prompt "A photo of scenery" \
--adapter_ckpt_path models/t2iadapter_color_sd14v1-7cb31ebd.ckpt \
--ddim \
--adapter_condition color \
--condition_image samples/color/color_0002.png \
--scale 9
```

#### Depth Adapter

<p align="center">
<img height="366" alt="Depth input" src="https://github.com/mindspore-lab/mindone/assets/16683750/592ee66b-d3cc-428b-8efc-f3f89a02e5bd"/>
<img height="366" alt="Depth output" src="https://github.com/mindspore-lab/mindone/assets/16683750/30efde92-eaaa-4e74-8c4f-c7fa8535bfc3"/>
<br><em>Prompt: desk, best quality, extremely detailed</em>
</p>

```shell
python examples/stable_diffusion_v2/adapter_image2image.py \
--version 1.5 \
--prompt "desk, best quality, extremely detailed" \
--adapter_ckpt_path models/t2iadapter_depth_sd15v2-dc86209b.ckpt \
--ddim \
--adapter_condition depth \
--condition_image samples/depth/desk_depth.png
```

#### OpenPose Adapter

<p align="center">
<img width="256" alt="OpenPose input" src="https://github.com/mindspore-lab/mindone/assets/16683750/04b8aa78-3914-46c0-bf8c-37acc064cd4d"/>
<img width="256" alt="OpenPose output" src="https://github.com/mindspore-lab/mindone/assets/16683750/56db6db4-aa77-4b15-ae03-485bb8144e71"/>
<br><em>Prompt: Iron man, high-quality, high-res</em>
</p>

```shell
python examples/stable_diffusion_v2/adapter_image2image.py \
--version 1.5 \
--prompt "Iron man, high-quality, high-res" \
--adapter_ckpt_path models/t2iadapter_openpose_sd14v1-ebcdb5cb.ckpt \
--ddim \
--adapter_condition openpose \
--condition_image samples/openpose/iron_man_pose.png
```

#### Segmentation Adapter

<p align="center">
<img width="320" alt="Segmentation input" src="https://github.com/mindspore-lab/mindone/assets/16683750/f7486cf2-1c0d-4b4f-bb04-fcfc1aeeae87"/>
<img width="320" alt="Segmentation output" src="https://github.com/mindspore-lab/mindone/assets/16683750/6e3b547d-1c42-4b4b-9751-26bd9d80aee4"/>
<br><em>Prompt: A black Honda motorcycle parked in front of a garage, best quality, extremely detailed</em>
</p>

```shell
python examples/stable_diffusion_v2/adapter_image2image.py \
--version 1.5 \
--prompt "A black Honda motorcycle parked in front of a garage, best quality, extremely detailed" \
--adapter_ckpt_path models/t2iadapter_seg_sd14v1-1d2e8478.ckpt \
--ddim \
--adapter_condition seg \
--condition_image samples/seg/motor.png
```

#### Sketch Adapter

<p align="center">
<img width="320" alt="Sketch input" src="https://github.com/mindspore-lab/mindone/assets/16683750/feef1c90-6ed9-4af4-a3ee-a949000fcc59"/>
<img width="320" alt="Sketch output" src="https://github.com/mindspore-lab/mindone/assets/16683750/e5661fe2-dc29-4f1b-a901-387329b06e52"/>
<br><em>Prompt: A car with flying wings</em>
</p>

```shell
python examples/stable_diffusion_v2/adapter_image2image.py \
--version 1.5 \
--prompt "A car with flying wings" \
--adapter_ckpt_path models/t2iadapter_sketch_sd15v2-6c537e26.ckpt \
--ddim \
--adapter_condition sketch \
--condition_image samples/sketch/car.png \
--cond_tau 0.5
```

### Combined Adapters

Individual T2I-Adapters can also be combined without retraining to condition on multiple images.

#### Color + Sketch

<p align="center">
<img height="256" alt="Sketch input" src="https://github.com/mindspore-lab/mindone/assets/16683750/a30661ce-26e3-42a3-b935-6caabe406d02"/>
<br><em>Prompt: A car with flying wings</em>
</p>

```shell
python examples/stable_diffusion_v2/adapter_image2image.py \
--version 1.5 \
--prompt "A car with flying wings" \
--adapter_ckpt_path models/t2iadapter_sketch_sd15v2.ckpt models/t2iadapter_color_sd14v1.ckpt \
--adapter_condition sketch color\
--condition_image samples/sketch/car.png samples/color/color_0004.png \
--cond_weight 1.0 1.2 \
--ddim
```

## Training

Coming soon.

## Acknowledgements

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, Ying Shan, Xiaohu Qie. T2I-Adapter: Learning
Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models. arXiv:2302.08453, 2023.
