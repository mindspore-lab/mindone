# CogVideoX Factory 🧪

[Read in English](./README_en.md)

在 Ascend 硬件下对 Cog 系列视频模型进行微调以实现自定义视频生成 ⚡️📼

> 我们的开发和验证基于Ascend 910*硬件，相关环境如下：
> | mindspore  | ascend driver  |  firmware   | cann toolkit/kernel |
> |:----------:|:--------------:|:-----------:|:------------------:|
> |    2.4     |    24.1.RC2    | 7.5.0.1.129 |      8.0.RC3       |

<table align="center">
<tr>
  <td align="center"><video src="https://github.com/user-attachments/assets/aad07161-87cb-4784-9e6b-16d06581e3e5">您的浏览器不支持视频标签。</video></td>
</tr>
</table>

## 快速开始

克隆并安装此仓库, 并且确保安装了相关依赖
```shell
pip install -e .[training]
pip install -r requirements.txt
```

> [!TIP]
> 数据读取依赖第三方python库`decord`，PyPI仅提供特定环境下的预构建文件以供安装。对于某些环境，您需要从源码编译并安装`decord`库。以下是EulerOS下安装`decord`的一个例子（参考自examples/latte）：
>
> 1. 您需要先安装`ffmpeg 4`，参考自 https://ffmpeg.org/releases:
> ```
>     wget https://ffmpeg.org/releases/ffmpeg-4.0.1.tar.bz2
>     tar -xvf ffmpeg-4.0.1.tar.bz2
>     mv ffmpeg-4.0.1 ffmpeg
>     cd ffmpeg
>     ./configure --enable-shared  # --enable-shared is needed for sharing libavcodec with decord
>     make -j 64
>     make install
> ```
> 2. 安装 `decord`, 参考自 [dmlc/decord](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source):
> ```
>     git clone --recursive https://github.com/dmlc/decord
>     cd decord
>     if [ -d build ];then rm -r build;fi && mkdir build && cd build
>     cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
>     make -j 64
>     make install
>     cd ../python
>     python setup.py install --user
> ```
> 最后，注意将当前路径添加到Python的搜索路径下。

接着下载数据集：

```
# 安装 `huggingface_hub`
huggingface-cli download   --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset   --local-dir video-dataset-disney
```

然后启动 LoRA 微调进行文本到视频的生成（根据您的选择修改不同的超参数、数据集根目录以及其他配置选项）：

```
# 对 CogVideoX 模型进行文本到视频的 LoRA 微调
./train_text_to_video_lora.sh

# 对 CogVideoX 模型进行文本到视频的完整微调
./train_text_to_video_sft.sh

# 对 CogVideoX 模型进行图像到视频的 LoRA 微调
./train_image_to_video_lora.sh

```

假设您的 LoRA 已保存到本地，并且路径为 `/path/to/my-awesome-lora`，现在我们可以使用微调模型进行推理：

```
import mindspore
from mindone.diffusers import CogVideoXPipeline
from mindone.diffusers import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", mindspore_dtype=mindspore.bfloat16
)
+ pipe.load_lora_weights("/path/to/my-awesome-lora", adapter_name=["cogvideox-lora"])
+ pipe.set_adapters(["cogvideox-lora"], [1.0])

video = pipe("<my-awesome-prompt>")[0][0]
export_to_video(video, "output.mp4", fps=8)
```

以下我们提供了更多探索此仓库选项的额外部分。所有这些都旨在尽可能降低内存需求，使视频模型的微调变得更易于访问。

## 训练

在开始训练之前，请你检查是否按照[数据集规范](./assets/dataset_zh.md)准备好了数据集。 我们提供了适用于文本到视频 (text-to-video) 和图像到视频 (image-to-video) 生成的训练脚本，兼容 [CogVideoX 模型家族](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce)。训练可以通过 `train*.sh` 脚本启动，具体取决于你想要训练的任务。让我们以文本到视频的 LoRA 微调为例。

> [!TIP]
> 由于模型和框架的限制，对于训练我们暂时推荐分阶段的训练流程，即先通过[`prepare_dateset.sh`](./prepare_dataset.sh)预处理数据集，然后读取预处理后的数据集通过`train*.sh`进行正式训练。
>
> 在正式训练阶段，需要增加`--load_tensors`参数以支持预处理数据集。建议增加参数`--mindspore_mode=0`以进行静态图训练加速，在`train*.sh`里可通过设置参数`MINDSPORE_MODE=0`实现。
>
> 具体情况参见[与原仓的差异 & 功能限制](#与原仓的差异功能限制)

### 预处理数据

通过[`prepare_dateset.sh`](./prepare_dataset.sh)预处理数据。注意其中用到的预训练模型、分辨率、帧率、文本的`max_sequence_length`设置都应当与正式训练一致！

- 配置用于预处理prompts和videos的模型：
```shell
MODEL_ID="THUDM/CogVideoX-2b"
```

- 配置用于预处理数据的NPU数量：
```shell
NUM_NPUS=8
```

- 配置待处理数据集读取配置和输出路径：
```shell
DATA_ROOT="/path/to/my/datasets/video-dataset"
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="/path/to/my/datasets/preprocessed-dataset"
```

- 配置prompts和videos预处理的相关参数（注意必须与正式训练的配置一致）：
```shell
HEIGHT_BUCKETS="480"
WIDTH_BUCKETS="720"
FRAME_BUCKETS="49"
MAX_NUM_FRAMES="49"
MAX_SEQUENCE_LENGTH=226
TARGET_FPS=8
```

- 配置预处理流程的批量大小、指定计算的数据类型：
```shell
BATCH_SIZE=1
DTYPE=bf16
```
然后正式运行`prepare_dateset.sh`，输出预处理后的数据集至`OUTPUT_DIR`

### 正式训练

- 配置用于训练的NPU数量：`NUM_NPUS=8`

- 选择训练的超参数。让我们以学习率和优化器类型的超参数遍历为例：

  ```shell
  LEARNING_RATES=("1e-4" "1e-3")
  LR_SCHEDULES=("cosine_with_restarts")
  OPTIMIZERS=("adamw" "adam")
  MAX_TRAIN_STEPS=("3000")
  ```

- 配置混合精度、ZeRO和MindSpore JIT加速配置：
  ```shell
  MINDSPORE_MODE=0
  AMP_LEVEL=O2
  DEEPSPEED_ZERO_STAGE=2
  ```

- 指定**预处理后**的字幕和视频的绝对路径以及列/文件。

  ```shell
  DATA_ROOT="/path/to/my/datasets/preprocessed-dataset"
  CAPTION_COLUMN="prompts.txt"
  VIDEO_COLUMN="videos.txt"
  ```

- 运行实验，遍历不同的超参数：
    ```shell
    for learning_rate in "${LEARNING_RATES[@]}"; do
    for lr_schedule in "${LR_SCHEDULES[@]}"; do
      for optimizer in "${OPTIMIZERS[@]}"; do
        for steps in "${MAX_TRAIN_STEPS[@]}"; do
          output_dir="./cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

          cmd="$LAUNCHER training/cogvideox_text_to_video_lora.py \
            --pretrained_model_name_or_path $MODEL_PATH \
            --data_root $DATA_ROOT \
            --caption_column $CAPTION_COLUMN \
            --video_column $VIDEO_COLUMN \
            --id_token BW_STYLE \
            --height_buckets 480 \
            --width_buckets 720 \
            --frame_buckets 49 \
            --dataloader_num_workers 2 \
            --validation_prompt \"BW_STYLE A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions:::BW_STYLE A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance\" \
            --validation_prompt_separator ::: \
            --num_validation_videos 1 \
            --validation_epochs 10 \
            --seed 42 \
            --lora_rank 128 \
            --lora_alpha 128 \
            --mixed_precision bf16 \
            --output_dir $output_dir \
            --max_num_frames 49 \
            --train_batch_size 1 \
            --max_train_steps $steps \
            --checkpointing_steps 1000 \
            --gradient_accumulation_steps 1 \
            --gradient_checkpointing \
            --learning_rate $learning_rate \
            --lr_scheduler $lr_schedule \
            --lr_warmup_steps 400 \
            --lr_num_cycles 1 \
            --enable_slicing \
            --enable_tiling \
            --load_tensors \
            --optimizer $optimizer \
            --beta1 0.9 \
            --beta2 0.95 \
            --weight_decay 0.001 \
            --max_grad_norm 1.0 \
            --report_to tensorboard \
            --mindspore_mode $MINDSPORE_MODE \
            --amp_level $AMP_LEVEL \
            --load_tensors \
            $EXTRA_ARGS"

          echo "Running command: $cmd"
          eval $cmd
          echo -ne "-------------------- Finished executing script --------------------\n\n"
        done
      done
    done
  done
  ```

要了解不同参数的含义，你可以查看 [args](./training/args.py) 文件，或者使用 `--help` 运行训练脚本。


## 与原仓的差异&功能限制

### 训练脚本结构性差异
为适配MindSpore平台特性，我们进行了框架接口的等价替换、调整了原有训练脚本的代码结构、构造了训练功能的等价实现等。如果您有意了解更深的代码细节，可参见[这篇文档](https://gist.github.com/townwish4git/3a181a1884747dfbbe4b31107ec02166)。

### MindSpore特性
我们为训练脚本提供了一些参数接口，用以对MindSpore的上下文和部分训练特性等进行相关配置：
- `distributed`: 开启并配置分布式训练
- `mindspore_mode`: 动/静态图配置
- `jit_level`: 编译优化级别
- `amp_level`：混合精度配置
- `zero_stage`: ZeRO优化器并行配置

具体使用方式参见[`args.py`](./training/args.py)中的`_get_mindspore_args()`。

### 功能限制

当前训练脚本并不完全支持原仓代码的所有训练参数，详情参见[`args.py`](./training/args.py)中的`check_args()`。

其中一个主要的限制来自于CogVideoX模型中的[3D Causual VAE不支持静态图](https://gist.github.com/townwish4git/b6cd0d213b396eaedfb69b3abcd742da)，这导致我们**不支持静态图模式下VAE参与训练**，因此在静态图模式下必须提前进行数据预处理以获取VAE-latents/text-encoder-embeddings cache。
