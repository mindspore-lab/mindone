# T2V-Turbo

This repository provides a Mindspore implementation of [T2V-Turbo](https://github.com/Ji4chenLi/t2v-turbo) from the following papers.

**T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback**  
Jiachen Li, Weixi Feng, Tsu-Jui Fu, Xinyi Wang, Sugato Basu, Wenhu Chen, William Yang Wang

Paper: https://arxiv.org/abs/2405.18750

![T2V-Turbo](assets/teasers/v1-pipeline.jpg)


## 📌 Features

- [x] T2V-Turbo-VC2 Inference
- [x] T2V-Turbo-MS Inference
- [x] T2V-Turbo-VC2 Training

## 🏭 Requirements

The scripts have been tested on Ascend 910B chips under the following requirements:

| mindspore | ascend driver | firmware | cann toolkit/kernel |
| --------- | ------------- | -------- | ------------------- |
| 2.4.0  | 24.1.RC3 | 7.5.0.1.129 | CANN 8.0.RC3.beta1 |
| 2.3.1  | 24.1.RC2 | 7.3.0.1.231 |	CANN 8.0.RC2.beta1 |

#### Installation Tutorials

1. Install Mindspore>=2.3.1 according to the [official tutorials](https://www.mindspore.cn/install)
2. Ascend users please install the corresponding *CANN* in [community edition](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1) as well as the relevant driver and firmware packages in [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers/community), as stated in the [official document](https://www.mindspore.cn/install/#%E5%AE%89%E8%A3%85%E6%98%87%E8%85%BEai%E5%A4%84%E7%90%86%E5%99%A8%E9%85%8D%E5%A5%97%E8%BD%AF%E4%BB%B6%E5%8C%85).
3. Install the pacakges listed in requirements.txt with `pip install -r requirements.txt`


## 🎯 Model Checkpoints

|Model|Resolution|Checkpoints|
|:---------|:---------|:--------|
|T2V-Turbo (VC2)|320x512|[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2/blob/main/unet_lora.pt) |
|T2V-Turbo (MS)|256x256|[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-MS/blob/main/unet_lora.pt) |

> **_NOTE:_**  The LORA weights here are originally in PyTorch format, please follow the instructions in [Inference](#-inference) to convert the weights in Mindspore format.


## 🚀 Inference

### 1) To play with our **T2V-Turbo (VC2)**, please follow the steps below:

#### Option 1: Automatically preprare the weights by running the `predict.py` script

By running the following command:

```bash
python predict.py \
  --prompt "input prompt for video generation" \
  --num_inference_steps 4
```

The model weights will automatically downloaded and converted in mindspore format and saved to `./model_cache/` folder in the following structure:

```bash
├─model_cache
│  ├─t2v-vc2
│  │  ├─VideoCrafter2_model_ms.ckpt
│  │  ├─unet_lora.ckpt
│  ├─open_clip_vit_h_14-9bb07a10.ckpt
```

#### Option 2: Manually download and convert the weights, set the paths with argments

1. Download the checkpoint of `VideoCrafter2` from [here](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
2. Download the `unet_lora.pt` of our T2V-Turbo (VC2) [here](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2/blob/main/unet_lora.pt).
3. Download the checkpoint of `OpenCLIP` from [here](https://download.mindspore.cn/toolkits/mindone/videocomposer/model_weights/open_clip_vit_h_14-9bb07a10.ckpt) and place it under the folder of `./model_cache/`
4. **Convert** the checkpoints to Mindspore Version by running the following commands:

```bash
# convert VideoCarfter2 Model
python tools/convert_weights.py --source PATH-TO-VideoCrafter2-model.ckpt --target PATH-TO-VideoCrafter2-MODEL.ckpt --type vc2

# convert unet_lora.pt
python tools/convert_weights.py --source PATH-TO-unet_lora.pt --target PATH_TO_UNET_LORA.ckpt --type lora
```

5. Generate text-to-video via following command:
```bash
python predict.py \
  --unet_dir PATH_TO_UNET_LORA.ckpt \
  --base_model_dir PATH-TO-VideoCrafter2-MODEL.ckpt \
  --prompt "input prompt for video generation" \
  --num_inference_steps 4
```

### 2) To play with our T2V-Turbo (MS), please follow the steps below:

1. Download model weights of `ModelScope` from [here](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b)
2. Download the `unet_lora.pt` of our T2V-Turbo (MS) [here](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-MS/blob/main/unet_lora.pt).
3. **Convert** the `unet_lora.pt` using the following command:

```bash
# convert unet_lora.pt
python tools/convert_weights.py --source PATH-TO-unet_lora.pt --target PATH-TO-unet_lora.ckpt --type lora
```

4. Generate text-to-video via following command:
```bash
python predict_ms.py \
  --unet_dir PATH_TO_UNET_LORA.ckpt \
  --base_model_dir PATH_TO_ModelScope_MODEL_FOLDER \
  --prompt "input prompt for video generation"\
  --num_inference_steps 4
```

## 🏋️ Training

### T2V-Turbo
To train T2V-Turbo (VC2), first prepare the data and model as below
1. Download the model checkpoint of VideoCrafter2 [here](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt).
2. Prepare the [WebVid-10M](https://github.com/m-bain/webvid) data. Save in the `webdataset` format.
3. Download the [InternVid2 S2 Model](https://huggingface.co/OpenGVLab/InternVideo2-CLIP-1B-224p-f8) 
4. Download the [HPSv2.1](https://huggingface.co/xswu/HPSv2/blob/main/HPS_v2.1_compressed.pt)
5. **Convert** the checkpoints to Mindspore Version by running the following commands:

```bash
# convert VideoCarfter2 Model
python tools/convert_weights.py --source PATH-TO-VideoCrafter2-model.ckpt --target PATH-TO-VideoCrafter2-model-ms.ckpt --type vc2

# convert InternVid2-S2 Model
python tools/convert_weights.py --source PATH-TO-InternVid2-S2 --target PATH-TO-unet_lora.ckpt --type internvid

# convert HPSv2.1 Model
python tools/convert_weights.py --source PATH-TO-HPSv2.1.pt --target PATH-TO-unet_lora.ckpt --type hps
```

6. Set `--pretrained_model_path`, `--data_path`, `--csv_path` and `--image_rm_ckpt_dir`, `--video_rm_ckpt_dir` accordingly in `scripts/train_t2v_turbo_vc2.sh`.

Then run the following command:
```bash
# standalone training
bash scripts/train_t2v_turbo_vc2.sh

# parallel
bash scripts/train_t2v_turbo_vc2_parallel.sh
```

### Three-stage training

Set the path of model and data as shown above in the `scripts/train_t2v_turbo_vc2_stages.sh`
The training process is as follow:

```bash
# standalone
bash scripts/train_t2v_turbo_vc2_stages.sh

# standalone
bash scripts/train_t2v_turbo_vc2_stages_parallel.sh
```


## 📋 Benchmarking

Experiments are tested on Ascend 910B with mindpsore 2.4.0 under pynative mode.

### Inference Performance

| model name | method | cards | batch size | resolution | precision | scheduler | steps | jit level | s/step | video/s | recipe |
| :--------: | :----: | :---: | :--------: | :--------: | :-------: | :-------: | :---: | :-------: | :----: | :-----: | :----: |
| T2V-Turbo (VC2) | LORA | 1 | 1 | 16x320x512 | fp16 | ddim | 4 | O1 | | | [yaml](./configs/inference_t2v_512_v2.0.yaml) |
| T2V-Turbo (MS)  | LORA | 1 | 1 | 16x256x256 | fp16 | ddim | 4 | O1 | | | [yaml](./configs/inference_t2v_512_v2.0.yaml) |

### Training Performance

| model name | method | cards | batch size | resolution | precision | jit level | s/step | video/s |
| :--------: | :----: | :---: | :--------: | :--------: | :-------: | :-------: | :----: | :-----: |
| T2V-Turbo (VC2) | LORA | 1 | 1 | 8x320x512 | fp16 | O1 | | |