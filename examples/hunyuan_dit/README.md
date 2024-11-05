<!-- ## **HunyuanDiT** -->

# Make Hunyuan-DiT run on MindSpore

> Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding.
> We've tried to provide a completely consistent interface and usage with the [Tencent/HunyuanDiT](https://github.com/Tencent/HunyuanDiT).

## Requirements

| mindspore | ascend driver | firmware | cann tookit/kernel |
| :---:       |   :---:        | :---:      | :---:      |
| 2.3.1     |  24.1RC2      |7.3.0.1.231|   8.0.RC2.beta1   |

## Features

- HunyuanDiT with the following features
  - ✔ mT5 TextEncoder model inference.
  - ✔ Acceleration methods: flash attention, mixed precision, etc..
  - ✔ Hunyuan-DiT-v1.0, Hunyuan-DiT-v1.1, and Hunyuan-DiT-v1.2 inference.
  - ✔ Hunyuan-DiT-v1.2 training.
  - ✔ ControlNet inference.

### TODO
* [ ] EMA
* [ ] ControlNet training
* [ ] Enhance prompt

## Dependencies and Installation

Begin by cloning the repository:
```shell
git clone https://github.com/mindspore-lab/mindone
cd mindone
pip install .
```

Then cd in the example folder `examples/hunyuan_dit` and run
```bash
pip install -r requirements.txt
```

Then download the model using the following commands:

```shell
# Create a directory named 'ckpts' where the model will be saved, fulfilling the prerequisites for running the demo.
mkdir ckpts
# Use the huggingface-cli tool to download the model.
# The download time may vary from 10 minutes to 1 hour depending on network conditions.
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.2 --local-dir ./ckpts
```

Currently, MindONE/Hunyuan-DiT only supports loading checkpoints through `safetensors`, so when the loaded weight type is `bin`, conversion is  required, which can be done through [safetensors/convert](https://huggingface.co/spaces/safetensors/convert).

---

## Training

### Data Preparation

  Refer to the commands below to prepare the training data.

  1. Install dependencies
      ```shell
      # 1 Install dependencies
      cd mindone/examples/hunyuan_dit
      pip install -e ./IndexKits
     ```
  2. Data download
     ```shell
     # 2 Data download
     wget -O ./dataset/data_demo.zip https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip
     unzip ./dataset/data_demo.zip -d ./dataset
     mkdir ./dataset/porcelain/arrows ./dataset/porcelain/jsons
     ```
  3. Data conversion
     ```shell  
     # 3 Data conversion
     python ./hydit/data_loader/csv2arrow.py ./dataset/porcelain/csvfile/image_text.csv ./dataset/porcelain/arrows 1
     ```

  4. Create training data index file using YAML file.

     ```shell
      # Single Resolution Data Preparation
      idk base -c dataset/yamls/porcelain.yaml -t dataset/porcelain/jsons/porcelain.json

      # Multi Resolution Data Preparation  
      idk multireso -c dataset/yamls/porcelain_mt.yaml -t dataset/porcelain/jsons/porcelain_mt.json
      ```

  The directory structure for `porcelain` dataset is:

  ```shell
   cd ./dataset

   porcelain
      ├──images/  (image files)
      │  ├──0.png
      │  ├──1.png
      │  ├──......
      ├──csvfile/  (csv files containing text-image pairs)
      │  ├──image_text.csv
      ├──arrows/  (arrow files containing all necessary training data)
      │  ├──00000.arrow
      │  ├──00001.arrow
      │  ├──......
      ├──jsons/  (final training data index files which read data from arrow files during training)
      │  ├──porcelain.json
      │  ├──porcelain_mt.json
   ```

### Full-parameter Training
  ```shell
  # Single Resolution Training
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain.json
  ```

  After checkpoints are saved, you can use the following command to evaluate the model.
  ```shell
  # Inference
  # You should replace the 'log_EXP/xxx/checkpoints/final.ckpt' with your actual path.
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只可爱的哈士奇" --no-enhance --dit-weight log_EXP/xxx/checkpoints/final.ckpt --load-key module --use-fp16
  ```

## Inference

### Using Command Line

We provide several commands to quick start:

```shell
# Only Text-to-Image. Flash Attention mode
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --no-enhance --use-fp16

# Generate an image with other image sizes.
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --image-size 1280 768 --use-fp16
```
<details onclose>
More example prompts can be found in [example_prompts.txt](example_prompts.txt)

### Using Previous versions

* **Hunyuan-DiT <= v1.1**

```shell
# ============================== v1.1 ==============================
# Download the model
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.1 --local-dir ./HunyuanDiT-v1.1
# Inference with the model
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --model-root ./HunyuanDiT-v1.1 --use-style-cond --size-cond 1024 1024 --beta-end 0.03 --use-fp16

# ============================== v1.0 ==============================
# Download the model
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./HunyuanDiT-v1.0
# Inference with the model
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --model-root ./HunyuanDiT-v1.0 --use-style-cond --size-cond 1024 1024 --beta-end 0.03 --use-fp16
```
</details>

### ControlNet

```shell
cd mindone/examples/hunyuan_dit
# 1. Use the huggingface-cli tool to download the model.
huggingface-cli download Tencent-Hunyuan/HYDiT-ControlNet-v1.2 --local-dir ./ckpts/t2i/controlnet
huggingface-cli download Tencent-Hunyuan/Distillation-v1.2 ./pytorch_model_distill.pt --local-dir ./ckpts/t2i/model
# 2. Convert this model to Safetensors
# 3. Quick start
python sample_controlnet.py --infer-mode fa --no-enhance --load-key distill --infer-steps 50 --control-type canny --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/canny.jpg --control-weight 1.0 --use-fp16
```


## Performance

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

### Training
| model name    |cards|batch size |  resolution |precision|jit level |step/s |
|:---:          |:---:|:---:      |  :---:      |:---:|:---:|:---:|
|HunyuanDiT-v1.2|1    |1          |  1024x1024  |fp16|O1|0.62|
|HunyuanDiT-v1.2|1    |1          |  1024x1024  |fp32|O1|0.45|

### Inference

| model name    | cards    | batch size | resolution   | scheduler |    steps |  jit level|step/s |  
|:-------:      |:--------:|:-------:   |:-----------: |:--------------:|:------------:|:-------:|:---------:|
|HunyuanDiT-v1.0|1         |1           |1024x1024     |ddpm            |100|O0|2.90|
|HunyuanDiT-v1.1|1         |1           |1024x1024     |ddpm            |100|O0|2.91|
|HunyuanDiT-v1.2|1         |1           |1024x1024     |ddpm            |100|O0|2.89|
