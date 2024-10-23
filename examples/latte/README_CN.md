# Latte：用于视频生成的潜在扩散transformer

## 1. Latte 简介

Latte [<a href="#references">1</a>] 是一种新颖的潜在扩散transformer，专为视频生成而设计。它基于 DiT（一种用于图像生成的扩散transformer模型）构建。有关 DiT [<a href="#references">2</a>] 的介绍，请参见 [DiT 的 README](../dit/README_CN.md)。

Latte 首先使用 VAE（变分自编码器）将视频数据压缩到潜在空间，然后根据潜在编码提取空间-时间标记。与 DiT 类似，它堆叠多个transformer块来模拟潜在空间中的视频扩散。如何设计空间和时间块成为一个主要问题。

通过实验和分析，他们发现最佳实践是下图中的结构（a）。它交替堆叠空间块和时间块，轮流模拟空间注意力和时间注意力。

<p align="center">
  <img src="https://raw.githubusercontent.com/Vchitect/Latte/9ededbe590a5439b6e7013d00fbe30e6c9b674b8/visuals/architecture.svg" width=550 />
</p>
<p align="center">
  <em> 图 1. Latte 及其transformer块的结构。 [<a href="#references">1</a>] </em>
</p>

与 DiT 类似，Latte 支持无条件视频生成和类别标签条件视频生成。此外，它还支持根据文本标题生成视频。

## 2. 快速开始

本教程将介绍如何使用 MindONE 运行推理和训练实验。

本教程包括：
- [x] 预训练ckpt转换；
- [x] 使用预训练的 Latte ckpt进行无条件视频采样；
- [x] 在 Sky TimeLapse 数据集上训练无条件 Latte：支持（1）使用视频训练；和（2）使用嵌入缓存训练；
- [x] 混合精度：支持（1）Float16；（2）BFloat16（将 patch_embedder 设置为 "linear"）；
- [x] 独立训练和分布式训练。
- [ ] 文本到视频 Latte 推理和训练。

### 2.1 环境设置

#### 配套版本

| mindspore | ascend driver | firmware     | cann toolkit/kernel|
|:----------|:---           | :--          |:--|
| 2.3.1     | 24.1.RC2      | 7.3.0.1.231  | 8.0.RC2.beta1|

```
pip install -r requirements.txt
```

`decord` 是视频生成所必需的。如果环境中没有 `decord` 包，请尝试 `pip install eva-decord`。
```
1. 安装 ffmpeg 4, 参考 https://ffmpeg.org/releases
    wget wget https://ffmpeg.org/releases/ffmpeg-4.0.1.tar.bz2 --no-check-certificate
    tar -xvf ffmpeg-4.0.1.tar.bz2
    mv ffmpeg-4.0.1 ffmpeg
    cd ffmpeg
    ./configure --enable-shared         # --enable-shared 是必选项，为了 decord 共享 libavcodec 的编解码库
    make -j 64
    make install
2. 安装 decord, 参考 https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source
    git clone --recursive https://github.com/dmlc/decord
    cd decord
    if [ -d build ];then rm build;fi && mkdir build && cd build
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
    make -j 64
    make install
    cd ../python
    python3 setup.py install --user
```

### 2.2 预训练ckpt

我们参考 [Latte 官方仓库](https://github.com/Vchitect/Latte/tree/main) 下载ckpt。在 FaceForensics、SkyTimelapse、Taichi-HD 和 UCF101 (256x256) 上训练的ckpt文件可以从 [huggingface](https://huggingface.co/maxin-cn/Latte/tree/main) 下载。

下载 `{}.pt` 文件后，请将其放置在 `models/` 文件夹下，然后运行 `tools/latte_converter.py`。例如，要转换 `models/skytimelapse.pt`，您可以运行：
```bash
python tools/latte_converter.py --source models/skytimelapse.pt --target models/skytimelapse.ckpt
```

请同时从 [huggingface/stabilityai.co](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main) 下载 VAE ckpt，并通过运行以下命令进行转换：
```bash
python tools/vae_converter.py --source path/to/vae/ckpt --target models/sd-vae-ft-mse.ckpt
```

## 3. 采样

例如，要在 Ascend 设备上使用 `256x256` 图像大小运行 `skytimelapse.ckpt` 模型的推理，您可以使用：
```bash
python sample.py -c configs/inference/sky.yaml
```

实验在MindSpore 2.3.1（图模式）的Ascend 910*上进行验证：

| model name | cards | image size | method | steps | jit level | ckpt loading time | compile time | total sample time |
| :--------: | :---: | :--------: | :----: | :---: | :-------: | :---------------: | :----------: | :---------------: |
|   latte    |   1   |  256x256   |  ddpm  |  250  |    O2     |      19.72s       |   101.26s    |      537.31s      |

这里展示了一些生成结果的例子：
<table class="center">
    <tr style="line-height: 0">
    <td width=33% style="border: none; text-align: center">Example 1</td>
    <td width=33% style="border: none; text-align: center">Example 2</td>
    <td width=33% style="border: none; text-align: center">Example 3</td>
    </tr>
    <tr>
    <td width=33% style="border: none"><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/latte/sky/generated-0.gif" style="width:100%"></td>
    <td width=33% style="border: none"><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/latte/sky/generated-1.gif" style="width:100%"></td>
    <td width=33% style="border: none"><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/latte/sky/generated-2.gif" style="width:100%"></td>
    </tr>
</table>
<p align="center">
  <em> 图 2. 从 torch ckpt转换的预训练模型生成的视频。 </em>
</p>

## 4. 训练

### 4.1 使用视频训练

我们支持在 Sky Timelapse 数据集上训练 Latte 模型，可以从 https://github.com/weixiong-ur/mdgan 下载。

解压缩下载的文件后，您将获得一个名为 `sky_train/` 的文件夹，其中包含所有训练视频帧。文件夹结构类似于：
```
sky_train/
├── video_name_0/
|   ├── frame_id_0.jpg
|   ├── frame_id_0.jpg
|   └── ...
├── video_name_1/
└── ...
```

首先，编辑配置文件 `configs/training/data/sky_video.yaml`。将 `data_folder` 从 `""` 更改为 `sky_train/` 的绝对路径。

然后，您可以使用以下命令在 Ascend 设备上开始独立训练：
```bash
python train.py -c configs/training/sky_video.yaml
```

要在 GPU 设备上开始训练，只需在上述命令后添加 `--device_target GPU`。

默认训练配置是从零开始训练 Latte 模型。批量大小是 $5$，训练周期数是 $3000$，这大约对应于 900k 步。学习率是恒定值 $1e^{-4}$。模型在混合精度模式下训练。默认的 AMP 级别是 `O2`。更多细节请参见 `configs/training/sky_video.yaml`。

为了加速训练速度，我们默认在配置文件中使用 `dataset_sink_mode: True`。您也可以设置 `enable_flash_attention: True` 进一步加速训练速度。

训练完成后，ckpt将保存在 `output_dir/ckpt/` 下。要使用ckpt运行推理，请将 `configs/inference/sky.yaml` 中的 `checkpoint` 更改为ckpt的路径，然后运行 `python sample.py -c configs/inference/sky.yaml`。

训练周期数设置得较大以确保收敛。您可以在准备好时随时终止训练。例如，我们使用了训练了 $1700$ 周期（大约 $500k$ 步）的ckpt，并用它进行了推理。以下是一些生成的示例：
<table class="center">
    <tr style="line-height: 0">
    <td width=33% style="border: none; text-align: center">Example 1</td>
    <td width=33% style="border: none; text-align: center">Example 2</td>
    <td width=33% style="border: none; text-align: center">Example 3</td>
    </tr>
    <tr>
    <td width=33% style="border: none"><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/latte/sky/epochs-1700-generated-0.gif" style="width:100%"></td>
    <td width=33% style="border: none"><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/latte/sky/epochs-1700-generated-1.gif" style="width:100%"></td>
    <td width=33% style="border: none"><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/latte/sky/epochs-1700-generated-2.gif" style="width:100%"></td>
    </tr>
</table>
<p align="center">
  <em> 图 3. 训练了大约 1700 轮（500k 步）的 Latte 模型生成的视频。 </em>
</p>

### 4.2 使用嵌入缓存训练

我们可以通过在运行训练脚本之前缓存数据集的嵌入来加速训练速度。这需要三个步骤：

- **步骤 1**：将嵌入缓存到一个缓存文件夹。请参阅以下关于如何缓存嵌入的示例。这一步可能需要一些时间。

要为 Sky Timelapse 数据集缓存嵌入，请首先确保 `configs/training/sky_video.yaml` 中的 `data_path` 正确设置为名为 `sky_train/` 的文件夹。然后您可以开始使用以下命令保存嵌入：

```bash
python tools/embedding_cache.py --config configs/training/sky_video.yaml --cache_folder path/to/cache/folder --cache_file_type numpy
```

您也可以将 `cache_file_type` 更改为 `mindrecord` 以 `.mindrecord` 文件形式保存嵌入。

通常，我们建议使用 `mindrecord` 文件类型，因为它得到 `MindDataset` 的支持，可以更好地加速数据加载。然而，Sky Timelapse 数据集有额外的长视频。使用 `mindrecord` 文件缓存嵌入增加了超过 MindRecord 写入器最大页面大小的风险。因此，我们建议使用 `numpy` 文件。

嵌入缓存过程可能需要一些时间，具体取决于视频数据集的大小。在此过程中可能会抛出一些异常。如果抛出了意外的异常，程序将停止，嵌入缓存写入器的状态将显示在屏幕上：

```bash
Start Video Index: 0. # 要处理的视频索引的开始
Saving Attempts: 0: save 120 videos, failed 0 videos. # 保存的视频文件数量
```

在这种情况下，您可以从索引 $120$（索引从 0 开始）的视频恢复嵌入缓存。只需附加 `--resume_cache_index 120`，然后运行 `python tools/embedding_cache.py`。它将从第 $120$ 个视频开始缓存嵌入，并保存嵌入，而不会覆盖现有文件。

要查看更多用法，请使用 `python tools/embedding_cache.py -h`。

- **步骤 2**：将数据集配置文件的 `data_folder` 更改为当前缓存文件夹路径。

缓存嵌入后，编辑 `configs/training/data/sky_numpy_video.yaml`，并将 `data_folder` 更改为存储缓存嵌入的文件夹。

- **步骤 3**：运行训练脚本。

您可以使用以下命令开始使用 Sky TimeLapse 的缓存嵌入数据集进行训练：

```bash
python train.py -c configs/training/sky_numpy_video.yaml
```

请注意，在 `sky_numpy_video.yaml` 中，我们使用了大量帧 $128$ 和较小的采样步长 $1$，这与 `sky_video.yaml`（num_frames=16 和 stride=3）中的设置不同。嵌入缓存使我们能够训练 Latte 生成更多帧，具有更大的帧率。

由于内存限制，我们将本地批量大小设置为 $1$，并使用梯度累积步数 $4$。训练周期数为 $1000$，学习率为 $2e^{-5}$。总训练步数约为 1000k。

如果出现 OOM（内存不足），请在 `configs/training/sky_numpy_video.yaml` 中设置 `enable_flash_attention: True`。它可以减少内存成本，也可以加速训练速度。

### 4.3 分布式训练

对于`mindspore>=2.3.0`，建议使用msrun启动使用ImageNet数据集格式的分布式训练，使用命令如下：

```
msrun --worker_num=4 \
    --local_worker_num=4 \
    --bind_core=True \
    --log_dir=msrun_log \
    python train.py \
    -c path/to/configuration/file \
    --use_parallel True
```

以 4 卡分布式训练为例，您可以使用以下命令开始分布式训练：
```bash
mpirun -n 4 python train.py \
    -c path/to/configuration/file \
    --use_parallel True
```
其中配置文件可以从 `configs/training/` 文件夹中的 `.yaml` 文件中选择。

如果您有 Ascend 设备的RankTable，可以参考 `scripts/run_distributed_sky_numpy_video.sh`，并使用以下命令开始 4 卡分布式训练：
```bash
bash scripts/run_distributed_sky_numpy_video.sh path/to/rank/table 0 4
```

第一个数字 `0` 表示训练设备的起始索引，第二个数字 `4` 表示您要启动的分布式进程总数。

## 5. 评估

实验在MindSpore 2.3.1（图模式）的Ascend 910*上进行验证：

| model name | cards | image size | graph compile | batch size | num frames | recompute | dataset sink mode | embedding cache | jit level | per step time | train. imgs/s |
| :--------: | :---: | :--------: | :-----------: | :--------: | :--------: | :-------: | :---------------: | :-------------: | :-------: | :-----------: | :-----------: |
|   latte    |   1   |  256x256   |   6~8 mins    |     5      |     16     |    OFF    |        ON         |       OFF       |    O2     |     1.03s     |     77.67     |
|   latte    |   1   |  256x256   |   6~8 mins    |     1      |    128     |    ON     |        ON         |       ON        |    O2     |     1.21s     |    105.78     |
|   latte    |   4   |  256x256   |   6~8 mins    |     1      |    128     |    ON     |        ON         |       ON        |    O2     |     1.32s     |    387.87     |
|   latte    |   8   |  256x256   |   6~8 mins    |     1      |    128     |    ON     |        ON         |       ON        |    O2     |     1.31s     |    781.67     |
>环境：{Ascend芯片}-{MindSpore版本}。
>训练图像/秒：训练过程中每秒处理的图片数量。训练图像/秒 = 卡数 * 批量大小 * 帧数 / 每步时间。

# 参考

[1] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, Yu Qiao: Latte: Latent Diffusion Transformer for Video Generation. CoRR abs/2401.03048 (2024)

[2] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023
