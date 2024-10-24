# Offline Inference with SDXL

## Requirements

⚠️Note: MindSpore Lite applyed python3.7. Please prepare the environment for Python 3.7 before installing it.

⚠️Note: MindSpore and MindSpore Lite must be the same version.

| mindspore      | mindspore   |   ascend driver | firmware    | cann toolkit/kernel |
|:--------------:|:-------------:|:-------------:|:-----------:|:-------------------:|
| 2.2.10 |2.2.10 | 23.0.3        | 7.1.0.5.220 | 7.0.0.beta1         |


Please refer to [Lite Install](https://mindspore.cn/lite/docs/zh-CN/r2.1/use/downloads.html).

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
  --task st_to_ms \
  --weight_safetensors /PATH TO/sd_xl_base_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_base_1.0_ms.ckpt \
  --key_torch torch_key_base.yaml \
  --key_ms mindspore_key_base.yaml
```

### 2. Export to MindSpore MindIR (`.ckpt` -> `.mindir`)

example as:

```shell
python export.py --task=text2img --model=./config/model/sd_xl_base_inference.yaml --n_samples=1
```

Note: The MindIR file will be generated in output/[MODEL_NAME]-[TASK].

### 3. Convert to MindSpore Lite Model (`.mindir` -> `_lite.mindir`)

Please use converter_lite command to convert MindSpore MindIR to the MindSpore Lite model, for example,

```shell script
converter_lite --fmk=MINDIR  --saveType=MINDIR --optimize=ascend_oriented \
  --modelFile=./output/[MODEL_NAME]-[TASK]/data_prepare_graph.mindir \
  --outputFile=./output/[MODEL_NAME]-[TASK]/data_prepare_graph_lite \
  --configFile=./config/lite/sd_lite.cfg
````

Note: Lite model name ends with `_lite.mindir`

## Offline Inference with MSLite

After all model conversions, run `sd_lite_infer.py` to generate images for the prompt of your interest, for example,

```shell
python sd_lite_infer.py --task=text2img --model=./config/model/sd_xl_base_inference.yaml \
  --sampler=./config/schedule/euler_edm.yaml --sampling_steps=40 --n_iter=1 --n_samples=1 --scale=9.0
```

Note: n_samples must be the same as the value in export.

## Performance

Experiments are tested on ascend 910* with mindspore 2.2.10 and MindSpore Lite 2.2.10. The sampler schedule is euler_edm.

| Model Name     | Task     | ImageSize | PerBatchSize | Sampler | Sample Step | Time Per Image |
|:---------------:|:--------:|:---------:|:------------:|:-----------:|:-----------:|:--------------:|
| sd_xl_base_1.0 | text2img | 1024*1024 | 1            | euler_edm| 40          | 6.172 s        |
