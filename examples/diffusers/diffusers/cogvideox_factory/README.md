# CogVideoX Factory 🧪

在 Ascend 硬件下对 Cog 系列视频模型进行微调以实现自定义视频生成 ⚡️📼

> 我们的开发和验证基于Ascend Atlas 800T A2硬件，相关环境如下：
> | mindspore  | ascend driver  |  firmware   | cann toolkit/kernel |
> |:----------:|:--------------:|:-----------:|:------------------:|
> |    2.5     |    24.1.RC2    | 7.5.0.1.129 |      8.0.0.beta1       |

<table align="center">
<tr>
  <td align="center"><video src="https://github.com/user-attachments/assets/aad07161-87cb-4784-9e6b-16d06581e3e5">您的浏览器不支持视频标签。</video></td>
</tr>
</table>

## 快速开始

克隆并安装此仓库, 并且确保安装了相关依赖
```shell
cd mindone
pip install -e .[training]
cd examples/diffusers/cogvideox_factory
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

然后启动 LoRA 或者SFT微调进行文本到视频的生成，详情参考[训练](#训练)：

```
# 对 CogVideoX 模型进行文本到视频的 LoRA 微调
./scripts/train_text_to_video_lora.sh

# 对 CogVideoX 模型进行文本到视频的完整微调
./scripts/train_text_to_video_sft.sh

# 对 CogVideoX 模型进行图像到视频的 LoRA 微调
./scripts/train_image_to_video_lora.sh

# 对 CogVideoX 模型进行图像到视频的完整微调
./scripts/train_image_to_video_sft.sh
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

以下是单卡、多卡推理和训练脚本说明。

## 推理

我们提供了脚本[`run_infer_text_to_video.sh`](./run_infer_text_to_video.sh)和[`run_infer_image_to_video.sh`](./run_infer_image_to_video.sh)用以执行单卡、多卡并行推理。

以`run_infer_text_to_video.sh`为例，

- 执行卡数及并行配置。注意当`SP=True`时，`MAX_SEQUENCE_LENGTH`必须是`SP_SIZE`的倍数，`SP_SIZE`不能是1：

```shell
NUM_NPUS=8
SP=True
SP_SIZE=$NUM_NPUS
DEEPSPEED_ZERO_STAGE=3
```

- 运行模式配置。`MINDSPORE_MODE=0`表示`graph mode`，`MINDSPORE_MODE=1`表示`pynative mode`，`JIT_LEVEL`表示在`graph mode`下加速的level：

```shell
MINDSPORE_MODE=0
JIT_LEVEL=O1
```

- 配置模型及推理结果参数。`MODEL_NAME_OR_PATH`默认是`THUDM/CogVideoX1.5-5b`，兼容[CogVideoX 模型家族](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce)，在联网环境会自动下载权重及配置文件，这里也能传入本地的权重及配置文件路径，结构需要和HuggingFace的CogVideoX 模型家族保持一致。`TRANSFORMER_PATH`和`LORA_PATH`可以不传，这时会使用`MODEL_NAME_OR_PATH`里的权重；`TRANSFORMER_PATH`和`LORA_PATH`配置需要二选一：

```shell
MODEL_NAME_OR_PATH="THUDM/CogVideoX1.5-5b"
# TRANSFORMER_PATH and LORA_PATH only choose one to set.
TRANSFORMER_PATH=""
LORA_PATH=""
PROMPT=""
H=768
W=1360
F=77
MAX_SEQUENCE_LENGTH=224
```

> [!TIP]
> H, W, F配置最好和训练保持一致；
> 开SP时，MAX_SEQUENCE_LENGTH必须是SP的倍数。

然后正式运行`run_infer_text_to_video.sh`，输出结果至`OUTPUT_DIR`。

## 训练

在开始训练之前，请你检查是否按照[数据集规范](./assets/dataset_zh.md)准备好了数据集。 我们提供了适用于文本到视频 (text-to-video) 生成和图像到视频 (image-to-video) 生成的训练脚本，兼容 [CogVideoX 模型家族](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce)。正式训练可以通过 `train*.sh` 脚本启动，具体取决于你想要训练的任务。让我们以文本到视频的 SFT 微调为例。

> [!TIP]
> 由于模型的特点：文本编码器及视频编码器只需要推理且文本编码器参数量较大，对于固定shape的训练我们推荐分阶段的训练流程，即先进行[预处理数据](#预处理数据)缓存数据，然后读取缓存通过`train*.sh`进行正式训练。
>
> 在正式训练阶段，需要增加`--embeddings_cache`参数以支持text embeddings预处理，`--vae_cache`参数以支持vae预处理。
>
> 具体情况参见[与原仓的差异 & 功能限制](#与原仓的差异功能限制)
>
> 多分辨率场景，推荐增加`--embeddings_cache`参数以支持text embeddings预处理，视频处理部分我们提供了一种分桶训练的方法，即将数据原始数据按实际数据的分辨率和帧数处理成一些设定好的BatchSize，分辨率和帧数。详情参见[启用分桶训练](#启用分桶训练)

### 预处理数据

通过[`prepare_dateset.sh`](./scripts/prepare_dataset.sh)预处理数据。注意其中用到的预训练模型、分辨率、帧率、文本的`max_sequence_length`设置都应当与正式训练一致！

- 配置用于预处理prompts和videos的模型：
```shell
MODEL_NAME_OR_PATH="THUDM/CogVideoX1.5-5b"
```

- 配置用于预处理数据的NPU数量：
```shell
NUM_NPUS=8
```

- 配置缓存数据，固定shape建议都缓存，多分辨率场景建议缓存`EMBEDDINGS_CACHE`。

```shell
VAE_CACHE=1
EMBEDDINGS_CACHE=1
```

- 当执行图像到视频的生成训练时，当`VAE_CACHE=1`时需要配置额外缓存图片数据

```shell
SAVE_IMAGE_LATENTS=1
```

- 配置待处理数据集读取配置和输出路径, `CAPTION_COLUMN`，`VIDEO_COLUMN`需要是`DATA_ROOT`实际prompt和video的文件路径，具体要求见[数据集规范](./assets/dataset_zh.md)：
```shell
DATA_ROOT="/path/to/my/datasets/video-dataset"
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="/path/to/my/datasets/preprocessed-dataset"
```

- 配置videos预处理的相关参数，`VAE_CACHE=1`时生效，注意必须与正式训练的配置一致：

```shell
H=768
W=1360
F=77
TARGET_FPS=8
```

- 配置prompts预处理的相关参数，`EMBEDDINGS_CACHE=1`时生效，注意必须与正式训练的配置一致：

```shell
MAX_SEQUENCE_LENGTH=224
```

- 配置预处理流程的批量大小、指定计算的数据类型：
```shell
BATCH_SIZE=1
DTYPE=bf16
```

然后正式运行`prepare_dateset.sh`，输出预处理后的数据集至`OUTPUT_DIR`

### 启用分桶训练

多分辨率场景，我们提供了一种分桶训练的方法，即将数据原始数据按实际数据的分辨率和帧数处理成一些设定好的BatchSize，分辨率和帧数。

训练时，在[`train_text_to_video_sft.sh`](./scripts/train_text_to_video_sft.sh)中配置`ENABLE_DYNAMIC_SHAPE=1`。分桶的配置文件需要使用`--bucket_config`传入，默认为[`./scripts/bucket.yaml`](./scripts/bucket.yaml)。

例如，配置支持480p和720p两种分辨率，bucket配置文件可以是：

```yaml
bucket_config:
  # Structure: "resolution": { num_frames: [ keep_prob, batch_size ] }
  # Setting [ keep_prob, batch_size ] to [ 0.0, 0 ] forces longer videos into smaller resolution buckets
  "480p": { 37: [0.4, 8], 53: [0.4, 3], 101: [0.3, 2], 197: [1.0, 1], 381: [1.0, 1]}
  "720p": { 37: [0.5, 2], 53: [0.2, 1] , 77: [0.4, 1] }
```

配置结构 `"resolution": { num_frames: [ keep_prob, batch_size ] }`,resolution是分辨率，具体shape可参考[`aspect.py`](cogvideox/datasets/aspect.py)；
keep_prob为视频满足该分辨率和帧数要求下分配到该桶的概率；batch_size为训练时的batch_size。

该算法参考自[Open-Sora](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md#more-data-and-better-multi-stage-training)。

> [!WARNING]
> 由于MindSpore的bug，开启分桶训练暂不能使用SP。如需试用，请安装使用MindSpore开发版[MindSpore master daily](https://repo.mindspore.cn/mindspore/mindspore/version/202503/20250311/master_20250311010111_d8f6bcc25ba2aa51d5d4e8a1a8aeab31b382435e_newest/)。

### 正式训练

- 执行卡数及并行配置。注意当`SP=True`时`MAX_SEQUENCE_LENGTH`必须是`SP_SIZE`的倍数，`SP_SIZE`不能是1：

```shell
NUM_NPUS=8
SP=True
SP_SIZE=$NUM_NPUS
```

- 多机训练配置，`MASTER_ADDR`是主节点的物理IP地址，默认是`127.0.0.1`，`NODE_RANK`是第几个节点，从0开始计数。

```shell
MASTER_ADDR="127.0.0.1"
NODE_RANK="0"
```

- 选择训练的超参数。让我们以学习率和优化器类型的超参数遍历为例：

  ```shell
  MIXED_PRECISION="bf16"
  LEARNING_RATES=("1e-5")
  LR_SCHEDULES=("cosine_with_restarts")
  OPTIMIZERS=("adamw_bf16")
  MAX_TRAIN_STEPS=("100000")
  ```

- 配置混合精度、ZeRO和MindSpore JIT加速配置：
  ```shell
  MINDSPORE_MODE=0
  JIT_LEVEL=O1
  AMP_LEVEL=O2
  DEEPSPEED_ZERO_STAGE=3
  ```

- 指定**预处理后**的字幕和视频的绝对路径以及列/文件。

  ```shell
  DATA_ROOT="/path/to/my/datasets/preprocessed-dataset"
  CAPTION_COLUMN="prompts.txt"
  VIDEO_COLUMN="videos.txt"
  ```

- 配置模型：`MODEL_NAME_OR_PATH`默认是`THUDM/CogVideoX1.5-5b`，兼容[CogVideoX 模型家族](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce)，在联网环境会自动下载权重及配置文件，这里也能传入本地的权重及配置文件路径，结构需要和HuggingFace的CogVideoX 模型家族保持一致。

  ```shell
  MODEL_NAME_OR_PATH="THUDM/CogVideoX1.5-5b"
  ```

- 动态shape配置，默认使用[`./scripts/bucket.yaml`](./scripts/bucket.yaml)分桶配置：

  ```shell
  ENABLE_DYNAMIC_SHAPE=0
  ```

- 视频和文本输入配置，注意ENABLE_DYNAMIC_SHAPE=1时HWF配置不生效，使用[`./scripts/bucket.yaml`](./scripts/bucket.yaml)中分桶配置；当`SP=True`时`MAX_SEQUENCE_LENGTH`必须是`SP_SIZE`的倍数：

  ```shell
  H=768
  W=1360
  F=77
  MAX_SEQUENCE_LENGTH=224
  ```

- 是否使用数据缓存,推荐都打开：

  ```shell
  VAE_CACHE=1
  EMBEDDINGS_CACHE=1
  ```

  - 运行实验，遍历不同的超参数：
  ```shell
  for learning_rate in "${LEARNING_RATES[@]}"; do
    for lr_schedule in "${LR_SCHEDULES[@]}"; do
      for optimizer in "${OPTIMIZERS[@]}"; do
        for steps in "${MAX_TRAIN_STEPS[@]}"; do
          output_dir="${OUTPUT_ROOT_DIR}/cogvideox-sft__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

          cmd="$LAUNCHER ${SCRIPT_DIR}/cogvideox_text_to_video_sft.py \
            --pretrained_model_name_or_path $MODEL_NAME_OR_PATH \
            --data_root $DATA_ROOT \
            --caption_column $CAPTION_COLUMN \
            --video_column $VIDEO_COLUMN \
            --height_buckets $H \
            --width_buckets $W \
            --frame_buckets $F \
            --max_num_frames $F \
            --max_sequence_length=$MAX_SEQUENCE_LENGTH \
            --gradient_accumulation_steps 1 \
            --dataloader_num_workers 2 \
            --seed 42 \
            --mixed_precision $MIXED_PRECISION \
            --output_dir $output_dir \
            --train_batch_size 1 \
            --max_train_steps $steps \
            --checkpointing_steps 2000 \
            --gradient_checkpointing \
            --fa_gradient_checkpointing=$FA_RCP \
            --scale_lr \
            --learning_rate $learning_rate \
            --lr_scheduler $lr_schedule \
            --lr_warmup_steps 800 \
            --lr_num_cycles 1 \
            --enable_slicing \
            --enable_tiling \
            --optimizer $optimizer \
            --beta1 0.9 \
            --beta2 0.95 \
            --weight_decay 0.001 \
            --max_grad_norm 1.0 \
            --report_to tensorboard \
            --mindspore_mode $MINDSPORE_MODE \
            --jit_level $JIT_LEVEL \
            --amp_level $AMP_LEVEL \
            --enable_sequence_parallelism $SP \
            --sequence_parallel_shards $SP_SIZE \
            $EXTRA_ARGS"

          echo "Running command: $cmd"
          eval $cmd
          echo -ne "-------------------- Finished executing script --------------------\n\n"
        done
      done
    done
  done
  ```

> [!TIP]
> 如果想修改transformer的模型结构，可以设置`--transformer_config`。比如修改成30B的模型，可以设置`--transformer_config=configs/cogvideox1.5_30B.yaml`；
> 当配置了`transformer_config`，可以配置`--transformer_ckpt_path`加载checkpoint权重。

要了解更多参数的含义，你可以查看 [args](./scripts/args.py) 文件，或者使用 `--help` 运行训练脚本。

## 性能数据

### 训练

|       model       | cards | DP | SP | zero  | vae cache | video shape | precision | jit level | s/step | memory usage |
|:-----------------:|:-----:|:--:|:--:|:-----:|:---------:|:-----------:|:---------:|:---------:|:------:|:------------:|
| CogvideoX 1.5 T2V 5B  |   8   | 8  | 1  | zero3 |    ON     | 1x77x768x1360 |   bf16    |    O1     | 39.23  |   35.6 GB    |
| CogvideoX 1.5 T2V 5B  |   8   | 4  | 2  | zero3 |    ON     | 1x77x768x1360 |   bf16    |    O1     |  20.9  |   19.9 GB    |
| CogvideoX 1.5 T2V 5B  |   8   | 2  | 4  | zero3 |    ON     | 1x77x768x1360 |   bf16    |    O1     |  10.1  |   14.6 GB    |
| CogvideoX 1.5 T2V 5B  |   8   | 1  | 8  | zero3 |    ON     | 1x77x768x1360 |   bf16    |    O1     |  5.16  |    8.2 GB    |
| CogvideoX 1.5 T2V 5B  |  16   | 2  | 8  | zero3 |    ON     | 1x77x768x1360 |   bf16    |    O1     |  5.24  |    6.3 GB    |
| CogvideoX 1.5 T2V 5B  |   8   | 8  | 1  | zero3 |    OFF    | 1x77x768x1360 |   bf16    |    O1     |   49   |    40 GB     |
| CogvideoX 1.5 T2V 5B  |   8   | 1  | 8  | zero3 |    OFF    | 1x77x768x1360 |   bf16    |    O1     | 10.58  |    9.3 GB    |
| CogvideoX 1.5 T2V 10B |   8   | 2  | 4  | zero3 |    ON     | 1x77x768x1360 |   bf16    |    O1     |  15.2  |   25.6 GB    |
| CogvideoX 1.5 T2V 20B |   8   | 2  | 4  | zero3 |    ON     | 1x77x768x1360 |   bf16    |    O1     |  20.1  |   35.7 GB    |
| CogvideoX 1.5 T2V 30B |   8   | 2  | 4  | zero3 |    ON     | 1x77x768x1360 |   bf16    |    O1     |  26.5  |   47.3 GB    |

以上数据在Disney数据集，Ascend Atlas 800T A2训练服务器上获得。

### 推理

|       model       | cards | DP | SP | zero  |  video shape  | precision | jit level | s/step | total cost |
|:-----------------:|:-----:|:--:|:--:|:-----:|:-------------:|:---------:|:---------:|:------:|:----------:|
| CogvideoX 1.5 T2V 5B  |   8   | 1  | 8  | zero3 | 1x77x768x1360 |   bf16    |    O1     |  3.21  |   ~ 5min   |

以上数据在Ascend Atlas 800T A2训练服务器上获得。

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

具体使用方式参见[`args.py`](./scripts/args.py)中的`_get_mindspore_args()`。

### 功能限制

当前训练脚本并不完全支持原仓代码的所有训练参数，详情参见[`args.py`](./scripts/args.py)中的`check_args()`。

其中一个主要的限制来自于CogVideoX模型中的[3D Causual VAE不支持静态图](https://gist.github.com/townwish4git/b6cd0d213b396eaedfb69b3abcd742da)，这导致我们**不支持静态图模式下VAE参与训练**，因此在静态图模式下必须提前进行数据预处理以获取VAE-latents/text-encoder-embeddings cache。
