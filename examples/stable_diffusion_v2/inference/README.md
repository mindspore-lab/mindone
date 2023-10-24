# Stable Diffusion Inference

## Installation Guide

<details close>

<summary> Details of Installation Guide </summary>>

Note: MindSpore Lite applyed python3.7. Please prepare the environment for Python 3.7 before installing.

### Install MindSpore

Please install MindSpore 2.1 refer to [MindSpore Install](https://www.mindspore.cn/install)

### Install MindSpore Lite

Refer to [Lite install](https://mindspore.cn/lite/docs/zh-CN/r2.1/use/downloads.html)

1. Download the supporting tar.gz and whl packages according to the environment.
2. Unzip the tar.gz package and install the corresponding version of the WHL package.

   ```shell
   tar -zxvf mindspore-lite-2.1.0-*.tar.gz
   pip install mindspore_lite-2.1.0-*.whl
   ```

3. Configure Lite's environment variables

   `LITE_HOME` is the folder path extracted from tar.gz, and it is recommended to use an absolute path.

   ```shell
   export LITE_HOME=/path/to/mindspore-lite-{version}-{os}-{platform}
   export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
   export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
   ```
**Note: MindSpore and MindSpore Lite must be same version.**

</details>

## Pretrained Weights

<details close>
  <summary>Pre-trained SD weights that are compatible with MindSpore: </summary>

Currently, we provide pre-trained stable diffusion model weights that are compatible with MindSpore as follows.

| **Version name** |**Task** |  **MindSpore Checkpoint**  | **Ref. Official Model** | **config** | **Resolution** |
|-----------------|---------------|---------------|------------|--------| ---- |
| 2.0            | text2img | [sd_v2_base-57526ee4.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt) |  [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base) | [v2-inference](config/model/v2-inference.yaml) | 512x512 |
| 2.0-v768      | text2img | [sd_v2_768_v-e12e3a9b.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_768_v-e12e3a9b.ckpt) |  [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) | [v2-inference](config/model/v2-inference.yaml) | 768x768 |
| 2.0-inpaint      | image inpainting | [sd_v2_inpaint-f694d5cf.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_inpaint-f694d5cf.ckpt) | [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) | [v2-inpaint-inference](config/model/v2-inpaint-inference.yaml) | 512x512 |
| 1.5       | text2img | [sd_v1.5-d0ab7146.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt) | [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | [v1-inference](config/model/v1-inference.yaml) | 512x512 |
| wukong    | text2img |  [wukong-huahua-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) |  | [v1-inference-chinese](config/model/v1-inference-chinese.yaml) | 512x512 |
| wukong-inpaint    | image inpainting |  [wukong-huahua-inpaint-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-inpaint-ms.ckpt) |  | [v1-inpaint-inference-chinese](config/model/v1-inpaint-inference-chinese.yaml) | 512x512 |
| controlnet-canny    | image inpainting |  [control_canny_sd_v1.5_static-6350d204.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/control_canny_sd_v1.5_static-6350d204.ckpt) | [controlnet-canny](https://github.com/lllyasviel/ControlNet/blob/main/gradio_canny2image.py) | [v1-controlnet-canny](config/model/v1-inference-controlnet.yaml) | 512x512 |
| controlnet-segmentation    | image inpainting |  [control_segmentation_sd_v1.5_static-77bea2e9.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/control_segmentation_sd_v1.5_static-77bea2e9.ckpt) | [controlnet-segmentation](https://github.com/lllyasviel/ControlNet/blob/main/gradio_canny2image.py) | [v1-controlnet-segmentation](config/model/v1-inference-controlnet.yaml) | 512x512 |

</details>

Please download the pretrained checkpoint and put it under `models/` folder.

To transfer other Stable Diffusion models to MindSpore, please refer to [model conversion](../tools/model_conversion/README.md).

## Support

### Device Inference Mode Support

for MindSpore2.1

| Device | Online Inference (MindSpore) | Offline Inference (Lite) |
| ------ | ---------------------------- | ------------------------ |
| Ascend 910A | ✅ | ✅ |
| Ascend 910B | - | ✅ |
| Ascend 310P | - | ✅ |
| GPU | ✅ | - |
| CPU | ✅ | - |

## Online Inference

Run `sd_infer.py` to generate images for the prompt of your interest.

```shell
python sd_infer.py --device_target=Ascend --task=text2img --model=./config/model/v2-inference.yaml --sampler=./config/schedule/ddim.yaml --sampling_steps=50 --n_iter=5 --n_samples=1 --scale=9.0
```

- device_target: Device target, should be in [Ascend, GPU, CPU], default is Ascend.
- task: Task name, should be [text2img, img2img, inpaint], if choose a task name, use the config/[task].yaml for inputs, default is text2img.
- model: Path to config which constructs model. Must be set, you can select a yaml from ./inference/config/model.
- sampler: Infer sampler yaml path, default is ./config/schedule/ddim.yaml.
- sampling_steps: Number of sampling steps, default is 50.
- n_iter: Number of iterations or trials, default is 1.
- n_samples: How many samples to produce for each given prompt in an iteration. A.k.a. batch size, default is 1.
- scale: Unconditional guidance scale. General set 7.5 for v1.x, 9.0 for v2.x

The checkpoint_path is set in model config path. If use lora pretrained checkpoint, please add those commands:

```shell
--use_lora=True --lora_rank=[LORA_RANK] --lora_ckpt_path=[LORA_CKPT_PATH]
```

Please run `python sd_infer.py -h` for details of command parameters.

The `prompt`, `negative_prompt`, `image_path`, generate image height, generate image width, could be set in **config/[task].yaml.**
If use Chinese model wukong, please modify `prompt`, `negative_prompt` to Chinese.

You can get images at "output/samples".

## Offline Inference

### Export

```shell
python export.py --task=text2img --model=./config/model/v2-inference.yaml --sampler=./config/schedule/ddim.yaml --n_samples=1
```

If use lora pretrained checkpoint, please add those commands:

```shell
--use_lora=True --lora_rank=[LORA_RANK] --lora_ckpt_path=[LORA_CKPT_PATH]
```

Please run `python export.py -h` for details of command parameters.

If only export MindSpore MindIR, please add `--converte_lite=False`.
If it already exists MindSpore MindIR, only need export MindSpore Lite MindIR, please add `--only_converte_lite=True`.
The MindSpore MindIR is common to different devices, but MindSpore Lite MindIR need to be re exported on different devices.

The MindIR file wil be generate in output/[MODEL_NAME]-[TASK]/, the MindSpore MindIR file end with .mindir without key word lite, the MindSpore Lite file end with _lite.mindir.
You can manually delete MindSpore MindIR files to save space.

### Lite Inference

Run `sd_lite_infer.py` to generate images for the prompt of your interest.

```shell
python sd_lite_infer.py --task=text2img --model=./config/model/v2-inference.yaml --sampler=./config/schedule/ddim.yaml --sampling_steps=50 --n_iter=5 --n_samples=1 --scale=9.0
```

Note: n_samples must be same as the value in export.

Please run `python sd_lite_infer.py -h` for details of command parameters.

You can get images at "output/samples".

## BenchMark

| Model | task | batch size | image size | sample step | device | engine | time per image |
| ----  | ---  | ---------- | ---------- | ----------- | ------ | ------ | -------------- |
| sd-2.0-base_fa | text2img | 1 | 512*512 | 50 | Ascend 910A | MindSpore | 5.49 s |
| sd-2.0-base-fa | text2img | 1 | 512*512 | 50 | Ascend 910A | Lite | 3.21 s |
| sd-2.0-base-fa | text2img | 1 | 512*512 | 50 | Ascend 910B | Lite | 2.7 s |

The sampler schedule is DDIM.
