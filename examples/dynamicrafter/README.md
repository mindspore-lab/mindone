# DynamiCrafter

This repository is the MindSpore implementation of [DynamiCrafter](https://arxiv.org/abs/2310.12190).


## Demo

We provide image to video generation with three resolutions: 256 (256\*256), 512 (320\*512), 1024 (576\*1024).

## Dependency

[MindSpore](https://www.mindspore.cn/install) 2.3

[CANN](https://repo.mindspore.cn/ascend/ascend910/20240705/) C18(0705)

```shell
pip install -r requirements.txt
```
## Inference

### Prepare prompts

Download the prompts from [here](https://download-mindspore.osinfra.cn/toolkits/mindone/dynamicrafter/prompts.zip) and then place them as directory `prompts/`.

### Prepare model weights

We provide weight conversion script `tools/convert_weight.py` to convert the original Pytorch model weights to MindSpore model weights. Pytorch model weights can be accessed via links below.

|Model|Resolution|Pytorch Checkpoint|
|:---------|:---------|:--------|
|DynamiCrafter1024|576x1024|[Hugging Face](https://huggingface.co/Doubiiu/DynamiCrafter_1024/blob/main/model.ckpt)|
|DynamiCrafter512|320x512|[Hugging Face](https://huggingface.co/Doubiiu/DynamiCrafter_512/blob/main/model.ckpt)|
|DynamiCrafter256|256x256|[Hugging Face](https://huggingface.co/Doubiiu/DynamiCrafter/blob/main/model.ckpt)|


The text files in `tools/` mark the model parameters mapping between Pytorch and MindSpore version. Select the ones according to the model you want to convert, and then run the following command to convert weight (e.g. 576\*1024).


```shell
cd tools
python convert_weight.py \
    --src_param ./pt_param_1024.txt \
    --target_param ./ms_param_1024.txt \
    --src_ckpt /path/to/pt/model_1024.ckpt \
    --target_ckpt /path/to/ms/model_1024.ckpt
```

### Run inference

```shell
sh scripts/run/run_infer.sh [RESUOUTION] [CKPT_PATH]
```

> [RESLLUTION] can be 256, 512 or 1024.

Inference speed on 910*:

|Model|Resolution|mode|jit_level|Speed(s/video)|
|:---------|:---------|:--------|:--------|:--------|
|DynamiCrafter1024|576x1024|GRAPH|O1|71|
|DynamiCrafter512|320x512|GRAPH|O1|21|
|DynamiCrafter256|256x256|GRAPH|O1|13|
