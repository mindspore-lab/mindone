# Tencent Hunyuan3D-1.0
> [Tencent Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation](https://arxiv.org/abs/2411.02293)
## **Abstract**
<p align="center">
  <img src="./assets/teaser.png"  height=450>
</p>

While 3D generative models have greatly improved artists' workflows, the existing diffusion models for 3D generation suffer from slow generation and poor generalization. To address this issue, we propose a two-stage approach named Hunyuan3D-1.0 including a lite version and a standard version, that both support text- and image-conditioned generation.

In the first stage, we employ a multi-view diffusion model that efficiently generates multi-view RGB in approximately 4 seconds. These multi-view images capture rich details of the 3D asset from different viewpoints, relaxing the tasks from single-view to multi-view reconstruction. In the second stage, we introduce a feed-forward reconstruction model that rapidly and faithfully reconstructs the 3D asset given the generated multi-view images in approximately 7 seconds. The reconstruction network learns to handle noises and in-consistency introduced by the multi-view diffusion and leverages the available information from the condition image to efficiently recover the 3D structure.

Our framework involves the text-to-image model, i.e., Hunyuan-DiT, making it a unified framework to support both text- and image-conditioned 3D generation. Our standard version has 3x more parameters than our lite and other existing model. Our Hunyuan3D-1.0 achieves an impressive balance between speed and quality, significantly reducing generation time while maintaining the quality and diversity of the produced assets.

## Updates
|Date| Features|
|---|---|
|12 December 2024| Support inference: text-to-mesh and image-to-mesh. <br> Individual modules include: <br> - (optional) text-to-image <br> - image background removal <br> - image-to-multiviews <br> - multiviews-to-mesh <br> -  (optional) mesh rendering (display device required)

## Get Started
### Install Environment
```
bash env_install.sh
```

Environment:
|mindspore |	Ascend driver | firmware | CANN tookit/kernel|
|--- | --- | --- | --- |
|2.3.1 | 24.1RC2 | 7.3.0.1.231 | 8.0.RC2.beta1|

### Download Pretrained Models

The models are available at [https://huggingface.co/tencent/Hunyuan3D-1](https://huggingface.co/tencent/Hunyuan3D-1):

+ `Hunyuan3D-1/lite`, lite model for multi-view generation.
+ `Hunyuan3D-1/std`, standard model for multi-view generation.
+ `Hunyuan3D-1/svrm`, sparse-view reconstruction model.

To download the model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python3 -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
mkdir weights
huggingface-cli download tencent/Hunyuan3D-1 --local-dir ./weights

mkdir weights/hunyuanDiT
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled --local-dir ./weights/hunyuanDiT
```

Refer to more details in [https://github.com/Tencent/Hunyuan3D-1](https://github.com/Tencent/Hunyuan3D-1/tree/main?tab=readme-ov-file#download-pretrained-models).


### Inference
For text to 3d generation, we supports bilingual Chinese and English, you can use the following command to inference.
```python
python3 main.py \
    --text_prompt "a lovely rabbit" \
    --save_folder ./outputs/test/ \
    --max_faces_num 90000
```

For image to 3d generation, you can use the following command to inference.
```python
python3 main.py \
    --image_prompt "/path/to/your/image" \
    --save_folder ./outputs/test/ \
    --max_faces_num 90000
```
You can also try prepared scripts with different configurations `scripts/text_to_3d_XX.sh` and  `scripts/image_to_3d_XX.sh`

We list some more useful configurations for easy usage:

|    Argument        |  Default  |                     Description                     |
|:------------------:|:---------:|:---------------------------------------------------:|
|`--text_prompt`  |   None    |The text prompt for 3D generation, support Chinese and English         |
|`--image_prompt` |   None    |The image prompt for 3D generation         |
|`--t2i_seed`     |    0      |The random seed for generating images        |
|`--t2i_steps`    |    25     |The number of steps for sampling of text to image  |
|`--gen_seed`     |    0      |The random seed for generating 3d generation        |
|`--gen_steps`    |    50     |The number of steps for sampling of 3d generation  |
|`--max_faces_numm` | 90000  |The limit number of faces of 3d mesh |
|`--do_render`  |   False   |render mesh into a gif in CPU (local display device required)  |
|`--use_lite`  |   False   | False to use std model, True to use lite model for multi-view generation. |

<!-- |`--save_memory`   | False   |module will move to cpu automatically|
|`--do_texture_mapping` |   False    |Change vertex shadding to texture shading  | -->

# Inference Performance
Tested on Ascend 910B1, MindSpore 2.3.1, Pynative mode.

|Model name| cards| batch size | resolution | steps| s/step |
|---|---|---|---|---|---|
|Hunyuan3D-1/lite| 1 | 1 | 512x512 | 50 |   1.6|
|Hunyuan3D-1/std | 1 | 1 | 512x512 | 50 |   3.2|
|Hunyuan3D-1/svrm| 1 | 1 | 7x512x512 | N.A. | 32|
