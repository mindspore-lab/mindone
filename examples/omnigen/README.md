<h1 align="center">OmniGen: Unified Image Generation</h1>


<p align="center">
    <a href="https://vectorspacelab.github.io/OmniGen/">
        <img alt="Build" src="https://img.shields.io/badge/Project%20Page-OmniGen-yellow">
    </a>
    <a href="https://arxiv.org/abs/2409.11340">
            <img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2409.11340-b31b1b.svg">
    </a>


</p>



## News
- **2025-03-20**: MindSpore implementation of OmniGen is released, supporting OmniGen inference and LoRA finetuning for T2I tasks.

## Overview

OmniGen is a unified image generation model that can generate a wide range of images from multi-modal prompts. It is designed to be simple, flexible, and easy to use.

![architecture](https://github.com/itruonghai/mindone-asset/blob/main/omnigen/architecture.png?raw=true)


## What Can OmniGen do?

OmniGen is a unified image generation model that you can use to perform various tasks, including but not limited to text-to-image generation, subject-driven generation, Identity-Preserving Generation, image editing, and image-conditioned generation. **OmniGen doesn't need additional plugins or operations, it can automatically identify the features (e.g., required object, human pose, depth mapping) in input images according to the text prompt.**

Here is the illustrations of OmniGen's capabilities:
![demo](https://github.com/itruonghai/mindone-asset/blob/main/omnigen/demo_cases.png?raw=true)


## 🏭 Requirements

The scripts have been tested on Ascend 910B chips under the following requirements:

| mindspore | ascend driver | firmware | cann toolkit/kernel |
| --------- | ------------- | -------- | ------------------- |
| 2.5  | 24.1.RC3 | 7.5.0.1.129 | CANN 8.0.RC3.beta1 |

#### Installation Tutorials

1. Install Mindspore>=2.4.10 according to the [official tutorials](https://www.mindspore.cn/install)
2. Ascend users please install the corresponding *CANN* in [community edition](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1) as well as the relevant driver and firmware packages in [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers/community), as stated in the [official document](https://www.mindspore.cn/install/#%E5%AE%89%E8%A3%85%E6%98%87%E8%85%BEai%E5%A4%84%E7%90%86%E5%99%A8%E9%85%8D%E5%A5%97%E8%BD%AF%E4%BB%B6%E5%8C%85).
3. Install the pacakges listed in requirements.txt with `pip install -r requirements.txt`


## Inference

### Model Download

You can download OmniGen model through [HuggingFace](https://huggingface.co/Shitao/OmniGen-v1).

```
huggingface-cli download Shitao/OmniGen-v1 --local-dir pretrained_model/
```

### Model Conversion

We need to convert the pytorch weight format to mindspore weight format. You can run this script

```bash
python omnigen/tools/omnigen_converter.py --source pretrained_model/model.safetensors --target ./models/omnigen.ckpt
```

You can run some examples of OmniGen on different tasks, by running
```bash
python infer.py
```

| input                                        | MS output                                                                                                                                    |
| :----------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `T2I`  <br /> "A vintage camera placed on the ground, ejecting a swirling  <br /> cloud of Polaroid-style photographs  into the air.  <br />The photos, showing landscapes, wildlife, and travel scenes,  <br /> seem to defy gravity as they float upwards."  | ![image](https://github.com/itruonghai/mindone-asset/blob/main/omnigen/docs_img/T2I_1.png?raw=true) |
|`T2I`  <br /> "A curly-haired man in <br /> a red shirt is drinking tea." | ![image](https://github.com/itruonghai/mindone-asset/blob/main/omnigen/docs_img/T2I_2.png?raw=true) |
|``Subject-driven Generation or Identity-Preserving Generation``  <br /> "The woman in *image_1* waves her hand happily in the crowd" <br /> <img src="https://github.com/itruonghai/mindone-asset/blob/main/omnigen/docs_img/ID_in_1.png?raw=true" width="200" height="200" />|![image](https://github.com/itruonghai/mindone-asset/blob/main/omnigen/docs_img/ID_out_1.png?raw=true) |
|``Subject-driven Generation or Identity-Preserving Generation``  <br /> "Two women are raising fried chicken legs in a bar. A woman is *image_1*. Another woman is *image_2*." <br /> <img src="https://github.com/itruonghai/mindone-asset/blob/main/omnigen/docs_img/ID_in_21.png?raw=true" width="200" height="200" /> <img src="https://github.com/itruonghai/mindone-asset/blob/main/omnigen/docs_img/ID_in_22.png?raw=true" width="400" height="200" />|![image](https://github.com/itruonghai/mindone-asset/blob/main/omnigen/docs_img/ID_out_2.png?raw=true) |


## LoRA finetuning

You can download the [toy dataset](https://github.com/VectorSpaceLab/OmniGen/tree/main/toy_data) of the original OmniGen implementation.

We provide a training script `train.py` to fine-tune OmniGen.
Here is a toy example about LoRA finetune:
```bash
python  train.py --model_name_or_path pretrained_model/  \
                 --batch_size_per_device 1  \
                 --mode 0 \
                 --condition_dropout_prob 0.01  \
                 --lr 1e-3  --use_lora  --lora_rank 8  \
                 --json_file ./toy_data/toy_subject_data.jsonl  \
                 --image_path ./toy_data/images  \
                 --max_input_length_limit 18000  \
                 --max_image_size 512  \
                 --gradient_accumulation_steps 1  \
                 --ckpt_every 50  --epochs 400 \
                 --log_every 1  --dtype bf16 \
                 --results_dir ./results/lora_training  \
                 --model_weight models/omnigen.ckpt \
```


## Performance

### Training
Experiments are tested on ascend 910* with mindspore 2.4.10, using `bfloat16`.

| model     | cards | recompute      | mode  | image size | attn  | batch size | step time (s/step) |
|---------------|:-------:|:-------:|:-----------:|:------------:|:------------:|:------------:|:--------------------:|
| OmniGen       | 1     | ✗| graph     | 512x512    | eager | 1                    | 0.33               |
| OmniGen       | 1     | ✓| graph     | 512x512    | eager | 1                    | 0.42              |
| OmniGen    | 1     | ✓|graph     | 1024x1024    | eager | 1                   | 3.2               |
### Inference
Experiments are tested on ascend 910* with mindspore 2.4.10 **pynative** mode, using `bfloat16` as floating point and enable `kv_cache` during inference. The subject-driven generation depends on number of input images that feed into the model.

| model     | task | cards   | image size |  step time (s/img) |
|---------------|:-------:|:-----------:|:------------:|:------------:|
| OmniGen       | Text to Image | 1     | 1024 x 1024    |  84s   |
| OmniGen       | Subject-driven generation | 1  |  1024 x 1024    |  150s (1 inputs) ~ 220s (2 inputs)  |

## License
This repo is licensed under the [MIT License](LICENSE).


## Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@article{xiao2024omnigen,
  title={Omnigen: Unified image generation},
  author={Xiao, Shitao and Wang, Yueze and Zhou, Junjie and Yuan, Huaying and Xing, Xingrun and Yan, Ruiran and Wang, Shuting and Huang, Tiejun and Liu, Zheng},
  journal={arXiv preprint arXiv:2409.11340},
  year={2024}
}
```
