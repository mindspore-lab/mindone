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
**[12 December 2024]** Support inference: text-to-mesh and image-to-mesh.

**- Features:** 
- (optional) text-to-image 
- image background removal 
- image-to-multiviews
- multiviews-to-mesh 
-  (optional) mesh rendering (display device required)


**- Comments:**
Differences from original [Hunyuan3D-1.0](https://github.com/Tencent/Hunyuan3D-1): 
- do not support texturing/backing. 
- use trimesh for mesh rendering in CPU, instead of Pytorch3D.

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



#### Inference
For text to 3d generation, it supports bilingual Chinese and English, you can use the following command to inference.
```python
text2image_path=Tencent-Hunyuan/HunyuanDiT-Diffusers
lite_pretrain=./weights/mvd_lite
mv23d_ckt_path=./weights/svrm/svrm.safetensors
python main.py \
    --text2image_path $text2image_path \
    --mvd_ckt_path $lite_pretrain \
    --mv23d_cfg_path ./svrm/configs/svrm.yaml \
    --mv23d_ckt_path $mv23d_ckt_path \
    --text_prompt "a lovely rabbit" \
    --save_folder ./outputs/test/ \
    --max_faces_num 90000 \
    --use_lite
```

For image to 3d generation, you can use the following command to inference.
```python
std_pretrain=./weights/mvd_std
mv23d_ckt_path=./weights/svrm/svrm.safetensors
python3 main.py \
    --mvd_ckt_path $std_pretrain \
    --mv23d_cfg_path ./svrm/configs/svrm.yaml \
    --mv23d_ckt_path $mv23d_ckt_path \
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

## Stage 1: Image to 6 Views
| model name|precision |  cards| batch size | resolution | jit level| flash attn| scheduler| steps| s/step |img/s|  weight|
|---|---|---|---|---|---|---|---|---|---|---|---|
|mvd_lite|fp16| 1 | 1 | 512x512 |O0| ON | euler ancestral discrete | 50 |   1.60| 0.075 | [weight](https://huggingface.co/tencent/Hunyuan3D-1/tree/main/mvd_lite)|
|mvd_std |fp16| 1 | 1 | 512x512 |O0| ON | euler ancestral discrete | 50 |   3.20| 0.038 | [weight](https://huggingface.co/tencent/Hunyuan3D-1/tree/main/mvd_std)|


\*note: checkpoint weights are originally float16. Flash attention uses bfloat16.

### Text/Image-to-views visual results
|Input | Lite | Std |
| --- | --- | --- |
|<img src="demos/example_003.png" style="width:256px"></img>|<img src="demos/mvd_output_example_003_lite.jpg" style="width:512px"></img>|<img src="demos/mvd_output_example_003_std.jpg" style="width:512px"></img>|
|<img src="demos/example_004.png" style="width:256px"></img>|<img src="demos/mvd_output_example_004_lite.jpg" style="width:512px"></img>|<img src="demos/mvd_output_example_004_std.jpg" style="width:512px"></img>|
|`一盆绿色植物生长在红色花盆中，居中，写实`|<img src="demos/mvd_output_text_002_lite.jpg" style="width:512px"></img>|<img src="demos/mvd_output_text_002_std.jpg" style="width:512px"></img>|
|`a lovely rabbit eating carrots`|<img src="demos/mvd_output_text_004_lite.jpg" style="width:512px"></img>|<img src="demos/mvd_output_text_004_std.jpg" style="width:512px"></img>|


<br>

## Stage 2: Views to Mesh

| model name|precision |  cards| batch size | resolution | jit level| flash attn| steps| s/step |mesh/s| recipe| weight|
|---|---|---|---|---|---|---|---|---|---|---|---|
|svrm |fp16| 1 | 1 | 7x512x512 |O0| ON| N/A | 32| 0.031|[svrm.yaml](./svrm/configs/svrm.yaml)| [weight](https://huggingface.co/tencent/Hunyuan3D-1/tree/main/svrm)|

\*note: checkpoint weights are originally float16. Flash attention always uses bfloat16, customized LayerNorm uses float32.

### Text/Image-to-mesh visual results
|Input | Lite | Std |
| --- | --- | --- |
|<img src="demos/example_003.png" style="width:256px"></img>|<img src="demos/svrm_output_example_003_lite.gif" style="width:512px"></img>|<img src="demos/svrm_output_example_003_std.gif" style="width:512px"></img>|
|<img src="demos/example_004.png" style="width:256px"></img>|<img src="demos/svrm_output_example_004_lite.gif" style="width:512px"></img>|<img src="demos/svrm_output_example_004_std.gif" style="width:512px"></img>|
|`一盆绿色植物生长在红色花盆中，居中，写实`|<img src="demos/svrm_output_text_002_lite.gif" style="width:512px"></img>|<img src="demos/svrm_output_text_002_std.gif" style="width:512px"></img>|
|`a lovely rabbit eating carrots`|<img src="demos/svrm_output_text_004_lite.gif" style="width:512px"></img>|<img src="demos/svrm_output_text_004_std.gif" style="width:512px"></img>|