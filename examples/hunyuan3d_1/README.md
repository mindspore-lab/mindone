# Tencent Hunyuan3D-1.0
> [Tencent Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation](https://arxiv.org/abs/2411.02293)
## **Introduction**
<p align="center">
  <img src="./assets/teaser.png"  height=450>
</p>

While 3D generative models have greatly improved artists' workflows, the existing diffusion models for 3D generation suffer from slow generation and poor generalization. To address this issue, Hunyuan3D-1.0 a two-stage approach named Hunyuan3D-1.0 including a lite version and a standard version, that both support text- and image-conditioned generation.
While 3D generative models have greatly improved artists' workflows, the existing diffusion models for 3D generation suffer from slow generation and poor generalization. Hunyuan3D-1.0, a two-stage approach, aims to address this issue. Hunyuan3D-1.0 includes a lite version and a standard version, that both support text- and image-conditioned generation.

In the first stage, Hunyuan3D-1.0 employs a multi-view diffusion model (`mvd-lite`/`mvd-std`) that efficiently generates multi-view RGB. These multi-view images capture rich details of the 3D asset from different viewpoints, relaxing the tasks from single-view to multi-view reconstruction. 

In the second stage, a feed-forward reconstruction model (`svrm`) rapidly and faithfully reconstructs the 3D asset given the generated multi-view images. The reconstruction network learns to handle noises and in-consistency introduced by the multi-view diffusion and leverages the available information from the condition image to efficiently recover the 3D structure.

The framework also involves the text-to-image model, i.e., [Hunyuan-DiT](https://github.com/chenyingshu/mindone/tree/master/examples/hunyuan_dit), making it a unified framework to support both text- and image-conditioned 3D generation. The standard version has 3x more parameters than the lite and other existing model. Hunyuan3D-1.0 achieves an impressive balance between speed and quality, significantly reducing generation time while maintaining the quality and diversity of the produced assets.

## Updates
|Date| Features|
|---|---|
|12 December 2024| Support inference: text-to-mesh and image-to-mesh. <br> Individual modules include: <br> - (optional) text-to-image <br> - image background removal <br> - image-to-multiviews <br> - multiviews-to-mesh <br> -  (optional) mesh rendering (display device required)

## Get Started
### Requirements
|mindspore |	ascend driver | firmware | cann tookit/kernel|
|--- | --- | --- | --- |
|2.3.1 | 24.1RC2 | 7.3.0.1.231 | 8.0.RC2.beta1|

### Dependencies
```
pip install -r requirements.txt
```

### Quick Start

#### Download Pretrained Models

The models are available at [https://huggingface.co/tencent/Hunyuan3D-1](https://huggingface.co/tencent/Hunyuan3D-1):

+ `Hunyuan3D-1/lite`, lite model for multi-view generation.
+ `Hunyuan3D-1/std`, standard model for multi-view generation.
+ `Hunyuan3D-1/svrm`, sparse-view reconstruction model.

<!-- To download the model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python3 -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
mkdir weights
huggingface-cli download tencent/Hunyuan3D-1 --local-dir ./weights

mkdir weights/hunyuanDiT
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled --local-dir ./weights/hunyuanDiT
``` -->



#### Inference
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


# Inference Performance
Experiments are tested on ascend 910* with mindSpore 2.3.1 pynative mode.

## Image to Views
| model name|precision |  cards| batch size | resolution | jit level| flash attn| scheduler| steps| s/step |img/s|  weight|
|---|---|---|---|---|---|---|---|---|---|---|---|
|mvd_lite|fp16| 1 | 1 | 512x512 |O0| ON | euler ancestral discrete | 50 |   1.60| 0.075 | [weight](https://huggingface.co/tencent/Hunyuan3D-1/tree/main/mvd_lite)|
|mvd_std |fp16| 1 | 1 | 512x512 |O0| ON | euler ancestral discrete | 50 |   3.20| 0.038 | [weight](https://huggingface.co/tencent/Hunyuan3D-1/tree/main/mvd_std)|


\*note: checkpoint weights are originally float16. Flash attention uses bfloat16.

### Image-to-views visual results

<br>

## Views to Mesh
| model name|precision |  cards| batch size | resolution | jit level| flash attn| steps| s/step |mesh/s| recipe| weight|
|---|---|---|---|---|---|---|---|---|---|---|---|
|svrm |fp16| 1 | 1 | 7x512x512 |O0| ON| N/A | 32| 0.031|[svrm.yaml](./svrm/configs/svrm.yaml)| [weight](https://huggingface.co/tencent/Hunyuan3D-1/tree/main/svrm)|

\*note: checkpoint weights are originally float16. Flash attention always uses bfloat16, customized LayerNorm uses float32.

### Image-to-mesh visual results