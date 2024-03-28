## DreamBooth为Stable Diffusion XL (SDXL)的微调

[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)

DreamBooth是一种用于个性化文本到图像扩散模型的方法，只需几张主体的图片（3~5）和其作为唯一标识符的名称。在微调过程中，会并行应用特定于类别的先验保留损失，利用模型对类别的语义先验，并鼓励输出的多样性。

例如，我们有一组特定[狗](https://github.com/google/dreambooth/tree/main/dataset/dog)的5张图片，属于微调的提示“一个 sks 狗”，其中“sks”是唯一标识符。同时，还会输入一般狗的图片，这些是文本提示“一个狗”中的类别图片，以便模型不会忘记其他狗的外观。

`train_dreambooth.py` 脚本在MindSpore和Ascend平台上实现了基于SDXL的DreamBooth微调。

**注意**：现在我们只允许通过[LoRA](https://arxiv.org/abs/2106.09685)对SDXL UNet进行DreamBooth微调。

## 准备

#### 依赖


确保已安装以下框架。

- mindspore 2.1.0（Ascend 910）/ mindspore 2.2.1（Ascend 910*）
- openmpi 4.0.3（用于分布式模式）

进入 `example/stable_diffusion_xl` 文件夹并运行


```shell
pip install -r requirement.txt
```

#### 预训练模型

从 huggingface 下载官方预训练权重，将权重从 `.safetensors` 格式转换为 Mindspore `.ckpt` 格式，并将它们放入 `./checkpoints/` 文件夹。请参阅 SDXL 的 [weight_convertion.md](./weight_convertion.md) 以获取详细步骤。

#### 微调数据集准备

微调数据集应该包含来自同一主题的 3-5 张图像，放置在同一个文件夹中。


```text
dir
├── img1.jpg
├── img2.jpg
├── img3.jpg
├── img4.jpg
└── img5.jpg
```

您可以在 [Google/DreamBooth](https://github.com/google/dreambooth/tree/main) 找到不同类别的图像。这里我们使用两个示例，[dog](https://github.com/google/dreambooth/tree/main/dataset/dog) 和 [dog6](https://github.com/google/dreambooth/tree/main/dataset/dog6)。它们显示如下，

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/961bdff6-f565-4cf2-85ce-e59c6ed547f3" width=800 />
</p>
<p align="center">
  <em> Figure 1. dog example: the five images from the subject dog for finetuning. </em>
</p>

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/a5bef2fc-b613-46de-8021-3e489dd663a1" width=800 />
</p>
<p align="center">
  <em> Figure 2. dog6 example: the five images from the subject dog for finetuning. </em>
</p>


## 微调

在运行微调脚本 `train_dreambooth.py` 之前，请指定可能因用户而异的参数。


* `--instance_data_path=/path/to/finetuning_data `
* `--class_data_path=/path/to/class_image `
* `--weight=/path/to/pretrained_model`
* `--save_path=/path/to/save_models`

如果需要，可以在运行命令时修改其他参数，或者在配置文件 `sd_xl_base_finetune_dreambooth_lora_910*.yaml` 中修改超参数。

使用多个NPU（例如4个）进行训练：


```shell
mpirun --allow-run-as-root -n 4 python train_dreambooth.py \
  --config configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --instance_data_path /path/to/finetuning_data \
  --instance_prompt "A photo of a sks dog" \
  --class_data_path /path/to/class_image \
  --class_prompt "A photo of a dog" \
  --ms_mode 0 \
  --save_ckpt_interval 500 \
  --is_parallel True \
  --device_target Ascend
```

使用以下命令启动独立训练：


```shell
python train_dreambooth.py \
  --config configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --instance_data_path /path/to/finetuning_data \
  --instance_prompt "A photo of a sks dog" \
  --class_data_path /path/to/class_image \
  --class_prompt "A photo of a dog" \
  --gradient_accumulation_steps 4 \
  --ms_mode 0 \
  --save_ckpt_interval 500 \
  --device_target Ascend
```

我们的实现使用了先验保护损失进行训练，可以避免过拟合和语言漂移。我们首先使用带有类别提示的预训练模型生成图像，并在微调过程中将这些数据与我们的数据并行输入。`train_dreambooth.py` 中的参数 `num_class_images` 指定了用于先验保护的类别图像数量。如果 `class_image_path` 中没有足够的图像，则会使用 `class_prompt` 进行额外的采样。当采样完成时，您需要使用上述命令重新启动训练。采样50张类别图像大约需要25分钟。

与Unet一起，SDXL中的两个文本编码器的训练也是支持的。实现方式也是通过 LoRA。要这样做，只需将上面的训练命令中的配置替换为 `configs/training/sd_xl_base_finetune_dreambooth_textencoder_lora_910b.yaml`。在我们的实验中，不使用文本编码器进行训练会产生更好的生成结果。

## 推理


请注意，上面的训练命令会在指定的 `save_path` 中获取微调后的 LoRA 权重。现在我们可以使用推理命令根据给定的提示生成图像。假设预训练的 ckpt 路径为 `checkpoints/sd_xl_base_1.0_ms.ckpt`，经过训练的 LoRA ckpt 路径为 `runs/SDXL_base_1.0_1000_lora.ckpt`，推理命令的示例如下。

* （推荐）使用交互式可视化运行。

  在 `demo/sampling.py` 中的常量 `VERSION2SPECS` 中替换权重和yaml文件的路径，在 `__main__` 中指定提示，然后运行：

  ```shell
  # (recommend) run with streamlit
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
  streamlit run demo/sampling.py --server.port <your_port>
  ```

* 使用另一个命令运行:

  ```shell
  # run with other commands
  python demo/sampling_without_streamlit.py \
    --task txt2img \
    --config configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml \
    --weight checkpoints/sd_xl_base_1.0_ms.ckpt,runs/SDXL_base_1.0_1000_lora.ckpt \
    --prompt "a sks dog swimming in a pool" \
    --device_target Ascend
  ```

关键字 `weight` 的两个权重（预训练权重和微调后的 LoRA 权重）之间用逗号分隔，没有空格。

使用不同提示生成的DreamBooth模型生成的图像示例如下。

[dog](https://github.com/google/dreambooth/tree/main/dataset/dog) 示例微调结果，


* "A photo of a sks dog swimming in a pool"
<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/0ddf4ce1-4177-44c0-84bd-2b15c0e2f6f4" width=700 />



[dog6](https://github.com/google/dreambooth/tree/main/dataset/dog6) 微调结果,

* "A photo of a sks dog in a bucket"

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/5144b904-329c-4d83-aa4b-c2f4ecd60ea0" width=700 />



* "A photo of a sks dog in a doghouse"

<p align="center">
  <img src="https://github.com/mindspore-lab/mindone/assets/33061146/6b2a6656-10a0-4d9d-8542-a9fa0527bc8a" width=700 />
