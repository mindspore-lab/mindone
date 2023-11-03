# Stable Diffusion XL Inference

## Installation Guide

⚠️Note: MindSpore Lite applyed python3.7. Please prepare the environment for Python 3.7 before installing it.
⚠️Note: MindSpore and MindSpore Lite must be the same version.

### 1. Install MindSpore

Please install MindSpore 2.1 refer to [MindSpore Install](https://www.mindspore.cn/install)

### 2. Install MindSpore Lite

Refer to [Lite Install](https://mindspore.cn/lite/docs/zh-CN/r2.1/use/downloads.html)

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

## Preparation

### 1. Convert Pretrained Weight (`.safetensors` -> `.ckpt`)

We provide a script for converting pre-trained weight from `.safetensors` to `.ckpt` in `tools/model_conversion/convert_weight.py`.

step1. Download the [Official](https://github.com/Stability-AI/generative-models) pre-train weights [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) from huggingface.

step2. Convert weight to MindSpore `.ckpt` format and put it to `./models/`.

```shell
# convert sdxl-base-1.0 model
cd tools/model_conversion
python convert_weight.py \
  --weight_safetensors /PATH TO/sd_xl_base_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_base_1.0_ms.ckpt \
  --key_torch torch_key_base.yaml \
  --key_ms mindspore_key_base.yaml
```

### 2. (For Offline Infer) Export to MindSpore MindIR (`.ckpt` -> `.mindir`)

example as:

```shell
python export.py --task=text2img --model=./config/model/sd_xl_base_inference.yaml --n_samples=1
```

Note: The MindIR file will be generated in output/[MODEL_NAME]-[TASK].

### 3. (For Offline Infer) Convert to MindSpore Lite Model (`.mindir` -> `_lite.mindir`)

Please use converter_lite command to convert MindSpore MindIR to the MindSpore Lite model, example as:

```shell script
converter_lite --fmk=MINDIR  --saveType=MINDIR --optimize=ascend_oriented \
  --modelFile=./output/[MODEL_NAME]-[TASK]/data_prepare_graph.mindir \
  --outputFile=./output/[MODEL_NAME]-[TASK]/data_prepare_graph_lite \
  --configFile=./config/lite/sd_lite.cfg
````

Note: Lite model name ends with `_lite.mindir`

## Offline Inference (MSLite)

After all model conversions, run `sd_lite_infer.py` to generate images for the prompt of your interest, example as:

```shell
python sd_lite_infer.py --task=text2img --model=./config/model/sd_xl_base_inference.yaml \
  --sampler=./config/schedule/euler_edm.yaml --sampling_steps=40 --n_iter=1 --n_samples=1 --scale=9.0
```

Note: n_samples must be the same as the value in export.

## Online Inference

<details close>

#### 1. Infer default (Recommend)

See [GATTING_STARTED](../GETTING_STARTED.md)

#### 2. Infer with the same process as MSLite (For debugging)

Run `sd_infer.py` to generate images for the prompt of your interest.

```shell
python sd_infer.py --device_target=Ascend --task=text2img --model=./config/model/sd_xl_base_inference.yaml --sampler=./config/schedule/euler_edm.yaml --sampling_steps=40 --n_iter=5 --n_samples=1 --scale=9.0
```

- device_target: Device target, default is Ascend.
- task: Task name, should be \[text2img\], if choose a task name, use the config/\[task\].yaml for inputs, default is text2img.
- model: Path to config which constructs the model. Must be set, you can select a yaml from ./inference/config/model.
- sampler: Infer sampler yaml path, default is ./config/schedule/euler_edm.yaml.
- sampling_steps: Number of sampling steps, default is 40.
- n_iter: Number of iterations or trials, default is 1.
- n_samples: How many samples to produce for each given prompt in an iteration. A.k.a. batch size, default is 1.
- scale: Unconditional guidance scale. General set 7.5 for v1.x, 9.0 for v2.x


Please run `python sd_infer.py -h` for details of command parameters.

The `prompt`, `negative_prompt`, and `image_path`, generate image height, generate image width, which could be set in **config/\[task\].yaml.**

You can get images at "output/samples".

</details>


## BenchMark (Offline Infer)

| Model | task | batch size | image size | sample step | device | engine | time per image |
| ----  | ---  | ---------- | ---------- | ----------- | ------ | ------ | -------------- |
| sd_xl_base_1.0 | text2img | 1 | 1024*1024 | 40 | Ascend 910A | MindSpore | 17.20 s |
| sd_xl_base_1.0 | text2img | 1 | 1024*1024 | 40 | Ascend 310P | MindSpore | 118 s |

The sampler schedule is euler_edm.


## Support

### Device Inference Mode Support

for MindSpore2.1

| Device | Online Inference (MindSpore) | Offline Inference (Lite) |
| ------ | ---------------------------- | ------------------------ |
| Ascend 910A | ✅ | ✅ |
| Ascend 310P | -  | ✅ |
