## 数据集格式

### 数据集结构

您的数据集结构应如下所示，dataset为数据集目录，里面包含提示词文件，视频路径文件和原始视频:

```
dataset
├── prompt.txt
├── videos.txt
├── videos
    ├── 00000.mp4
    ├── 00001.mp4
    ├── ...
```

### 提示词文件要求

创建 `prompt.txt` 文件，文件应包含逐行分隔的提示。请注意，提示必须是英文，并且建议使用 [提示润色脚本](https://github.com/THUDM/CogVideo/blob/main/inference/convert_demo.py) 进行润色。或者可以使用 [CogVideo-caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption) 进行数据标注：

```
A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.
A black and white animated sequence on a ship’s deck features a bulldog character, named Bully Bulldoger, showcasing exaggerated facial expressions and body language...
...
```

### 视频路径文件要求

支持的分辨率和帧数需要满足以下条件：

- **支持的分辨率（宽 * 高）**：
    - 任意分辨率且必须能被32整除。例如，`720 * 480`, `1920 * 1020` 等分辨率。

- **支持的帧数（Frames）**：
    - 必须是 `4 * k` 或 `4 * k + 1`（例如：16, 32, 49, 81）
    - CogvideoX 1.5版本，必须满足`((frame - 1) // 4 + 1) % 2 == 0`

所有的视频建议放在一个文件夹中。

接着，创建 `videos.txt` 文件，包含与`prompt.txt`对应的视频文件路径，请注意，视频文件路径需要是数据集目录的相对路径。格式如下：

```
videos/00000.mp4
videos/00001.mp4
...
```

对于有兴趣了解更多细节的开发者，您可以查看[`VideoDataset`](../cogvideox/dataset.py)。

### 数据集使用方法

数据集格式满足上面要求，训练时`--data_root`传入数据集目录路径；`--caption_column`传入数据集目录下提示词数据集路径，如`prompt.txt`；`--video_column`传入数据集目录下视频数据集路径，如`videos.txt`。

例如，使用 [这个](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset) Disney 数据集进行微调。下载可通过🤗
Hugging Face CLI 完成：

```
huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir video-dataset-disney
```

> 该数据集已按照预期格式准备好，可直接使用。

#### 数据预处理

训练时直接使用原始数据需要加载 [VAE](https://huggingface.co/THUDM/CogVideoX-5b/tree/main/vae)（将视频编码为潜在空间）和大型 [T5-XXL](https://huggingface.co/google/t5-v1_1-xxl/)（将文本编码）可能会导致OOM（内存不足）。

为了降低内存需求，可以使用 [`prepare_dataset.sh`](../scripts/prepare_dataset.sh) 脚本预先计算vae的encoder和文本编码器(text_encoder)的输出进行缓存。使用详情参考[预处理数据](../README.md#预处理数据)。
