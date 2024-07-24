# DynamiCrafter

This repository is the MindSpore implementation of [DynamiCrafter](https://arxiv.org/abs/2310.12190).


## Demo

We provide image to video generation with three resolutions: 256 (256\*256), 512 (320\*512), 1024 (576\*1024).

### 576*1024

| Input                                                                                                                                                                                                            | Output                                                                                                                 |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| <p align="center"><img width="300" src="https://github.com/user-attachments/assets/426d5f70-420a-49e2-a6e1-61cbe82e73ca"/><br/>"a man in an astronaut suit playing a guitar"</p> | <video width="300" src="https://github.com/user-attachments/assets/bb04f8f2-a6f9-4e19-b0b3-8d6f6735b13a"/> |
| <p align="center"><img width="300" src="https://github.com/user-attachments/assets/567e8cd3-1c6f-4054-a82e-db309d6dffe4"/><br/>"time-lapse of a blooming flower with leaves and a stem"</p> | <video width="300" src="https://github.com/user-attachments/assets/11c6cc86-152f-4589-a85e-40b58dd1d021"/> |
| <p align="center"><img width="300" src="https://github.com/user-attachments/assets/f9a762a8-ad18-4f00-b3b9-bae8eced7c0d"/><br/>"fireworks display"</p> | <video width="300" src="https://github.com/user-attachments/assets/149cd613-a2c7-488e-a674-eff04b4fbf93"/> |
| <p align="center"><img width="300" src="https://github.com/user-attachments/assets/f02c1f95-bd02-4ef7-9059-05919257eea5"/><br/>"a beautiful woman with long hair and a dress blowing in the wind"</p> | <video width="300" src="https://github.com/user-attachments/assets/dd2e7e92-085e-4432-af49-72619dd9ea4f"/> |

### 320*512

| Input                                                                                                                                                                                                            | Output                                                                                                                 |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| <p align="center"><img width="300" src="https://github.com/user-attachments/assets/f7f7f637-8811-4f90-8179-41f998fc4314"/><br/>"a bonfire is lit in the middle of a field"</p> | <video width="300" src="https://github.com/user-attachments/assets/c0020dae-ca26-4f0f-814c-fec9ac2513ee"/> |
| <p align="center"><img width="300" src="https://github.com/user-attachments/assets/13e75bfd-5bca-4250-8469-6f8dd8fede9c"/><br/>"a woman looking out in the rain"</p> | <video width="300" src="https://github.com/user-attachments/assets/294b19e4-a597-4d72-939b-a28361d9ad3c"/> |
| <p align="center"><img width="300" src="https://github.com/user-attachments/assets/8ef652da-9cae-4817-bbb0-67468a08f623"/><br/>"a sailboat sailing in rough seas with a dramatic sunset"</p> | <video width="300" src="https://github.com/user-attachments/assets/da1d0dc6-d1bf-4a4e-9011-3928778013fb"/> |
| <p align="center"><img width="300" src="https://github.com/user-attachments/assets/72f0a39c-69ee-472c-8916-f5566dbf1805"/><br/>"a group of penguins walking on a beach"</p> | <video width="300" src="https://github.com/user-attachments/assets/de878fd4-d399-47ed-bcee-9d10fc297464"/> |

### 256*256

| Input                                                                                                                                                                                                            | Output                                                                                                                 |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| <p align="center"><img width="200" src="https://github.com/user-attachments/assets/27970d71-04b6-43b5-ad83-4ece4c35bb3d"/><br/>"a brown bear is walking in a zoo enclosure, some rocks around"</p> | <video width="300" src="https://github.com/user-attachments/assets/6ef2baf7-66e6-4ef6-ad7f-806280ef7e10"/> |
| <p align="center"><img width="200" src="https://github.com/user-attachments/assets/14745684-40d3-4b4e-9920-b982bfb54761"/><br/>"two people dancing"</p> | <video width="300" src="https://github.com/user-attachments/assets/b1722f84-2e1b-4e46-940a-88c948d7293d"/> |
| <p align="center"><img width="200" src="https://github.com/user-attachments/assets/90ea568f-3980-4d5b-9ea8-43d3dc0bccec"/><br/>"a campfire on the beach and the ocean waves in the background"</p> | <video width="300" src="https://github.com/user-attachments/assets/e9c108a5-1f38-4486-ad8b-bfcd8b8eed96"/> |
| <p align="center"><img width="200" src="https://github.com/user-attachments/assets/a6d7951d-7d3f-4ddd-9225-6b71fe79ef7d"/><br/>"girl with fires and smoke on his head"</p> | <video width="300" src="https://github.com/user-attachments/assets/c3cb4ef9-681a-404c-a7cb-babf0a74acc3"/> |

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
