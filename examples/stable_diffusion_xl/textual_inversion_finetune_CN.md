# SDXL Textual Inversion微调方法

[一个图像就是一个词：使用Textual Inversion个性化文本到图像生成](https://arxiv.org/abs/2208.01618)

## 简介

<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/textual_inversion/textual_inversion_diagram.PNG" width=850 />
</p>
<p align="center">
  <em> Figure 1. The Diagram of Textual Inversion. [<a href="#references">1</a>] </em>
</p>


如上所示，Textual Inversion方法包括以下步骤：
1. 它创建一个文本提示，遵循某种模板，如 **一张$S_{*}$的照片**，其中$S_{*}$是待学习的新“词语”的占位符。
2. 分词器将为此占位符分配一个唯一的索引。此索引对应于嵌入查找表中的单个嵌入向量$v_{*}$，该表是可训练的，而所有其他参数都是不可训练的。
3. 在训练步骤中计算生成器（例如，稳定扩散模型）的损失函数和单个嵌入向量的梯度，然后在$v_{*}$中更新权重。

## 准备工作

#### 依赖


确保安装了以下框架。

- mindspore 2.1.0（Ascend 910）/ mindspore 2.2.1（Ascend 910*）
- openmpi 4.0.3（用于分布式模式）

进入 `example/stable_diffusion_xl` 文件夹并运行


```shell l
pip install -r requirement.txt
```

#### 预训练模型

从 HuggingFace 下载官方预训练权重，将权重从 `.safetensors` 格式转换为 Mindspore 的 `.ckpt` 格式，并将它们放到 `./checkpoints/` 文件夹中。请参考 SDXL [GETTING_STARTED.md](https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_xl/GETTING_STARTED.md#convert-pretrained-checkpoint) 获取详细步骤。

#### 微调数据集准备

根据我们希望微调的模型学习的概念，数据集可以分为两组：相同**对象**的数据集和相同**风格**的数据集。

对于**对象**数据集，我们使用 [cat-toy](https://huggingface.co/datasets/diffusers/cat_toy_example) 数据集。该数据集包含六张图片，如下所示。

<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/textual_inversion/cat-toy-examples.png" width=650 />
</p>
<p align="center">
  <em> Figure 2. The cat-toy example dataset for finetuning. </em>
</p>


对于**style**数据集，我们使用 [`chinese-art`](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets) 数据集的测试集，其中包含 20 张图片。以下是一些示例图片：

<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/textual_inversion/chinese_art_four_samples.png" width=650 />
</p>
<p align="center">
  <em> Figure 3. The example images from the test set of the chinese-art dataset </em>
</p>

有关下载 `chinese-art` 数据集的详细信息，请参阅 [LoRA: Text-image Dataset Preparation](https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_v2/README.md#dataset-preparation)。

同一数据集的微调图像应放置在同一文件夹下，如下所示：


```text
dir-to-images
├── img1.jpg
├── img2.jpg
├── img3.jpg
├── img4.jpg
└── img5.jpg
```
我们将包含对象数据集的文件夹命名为 `datasets/cat_toy`，将包含 `chinese_art` 数据集测试集的文件夹命名为 `datasets/chinese_art`。

## 微调

微调实验的关键参数解释如下：
- `num_vectors`: 文本编码器中可训练文本嵌入向量的数量。请注意，sd-xl 有两个文本编码器。较大的值表示较大的容量。
- `total_step`: 训练步数的总数。
- `gradient_accumulation_steps`: 梯度累积步数。当它等于一时，不会应用梯度累积。
- `placeholder_token`: 占位符 $S_{*}$。
- `learnable_property`: ["object", "style", "face"] 中的一个，表示学习目标为对象、风格还是面部特征。

### 对象数据集

cat-toy数据集的独立训练命令：

```bash
python train_textual_inversion.py \
    --data_path datasets/cat_toy \
    --save_path runs/cat_toy  \
    --infer_during_train True \
    --gradient_accumulation_steps 4 \
    --num_vectors 10 \
    --total_step 3000 \
    --placeholder_token "<cat-toy>"  \
    --learnable_property "object"
```


### Style数据集
chinese-art 数据集的独立训练命令：
```bash
python train_textual_inversion.py \
    --data_path datasets/chinese_art \
    --save_path runs/chinese_art  \
    --infer_during_train True \
    --gradient_accumulation_steps 4 \
    --num_vectors 2 \
    --total_step 2000 \
    --placeholder_token "<chinese-art>"  \
    --learnable_property "style"
```

**注意**：
1. 增加训练步数 `--total_step` 或可训练标记数量 `--num_vectors` 会增加过拟合的风险。
2. 默认情况下，我们对新的可学习标记嵌入使用随机初始化。但是，我们也允许使用现有标记的嵌入来初始化新的可学习标记嵌入。有关更多详细信息，请查看 `train_textual_inversion.py` 中的 `--initializer_token`。
3. 将 `--infer_during_train` 设置为 `True` 将使每个 `args.infer_interval` 步（默认为 500 步）进行一次推理。将其设置为 `False` 可节省一些训练时间。

## 推理

注意，上述训练命令在指定的 `save_path` 中获取了微调的文本反演权重。现在我们可以使用推理命令来根据给定的提示生成图像。假设预训练的 ckpt 路径是 `checkpoints/sd_xl_base_1.0_ms.ckpt`，训练的文本反演 ckpt 路径是 `runs/<dataset>/SD-XL-base-1.0_x_ti.ckpt`，推理命令的示例如下。

* 使用cat-toy学习到的embedding进行运行

  ```shell
  export MS_PYNATIVE_GE=1
  python demo/sampling_without_streamlit.py \
    --task txt2img \
    --config configs/training/sd_xl_base_finetune_textual_inversion.yaml \
    --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
    --textual_inversion_weight runs/cat_toy/SD-XL-base-1.0_3000_ti.ckpt \
    --prompt "a <cat-toy> backpack" \
    --device_target Ascend \
    --num_cols 4
  ```

* 使用 chinese-art 学习到的embedding进行运行

  ```shell
  export MS_PYNATIVE_GE=1
  python demo/sampling_without_streamlit.py \
    --task txt2img \
    --config configs/training/sd_xl_base_finetune_textual_inversion.yaml \
    --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
    --textual_inversion_weight runs/chinese_art/SD-XL-base-1.0_2000_ti.ckpt \
    --prompt "a dog in <chinese-art> style" \
    --device_target Ascend \
    --num_cols 4
  ```

建议使用交互式应用程序通过 streamlit 运行推理。请根据以下示例修改 `demo/sampling.py` 中的 `VERSION2SPECS`（注意修改 `config` 和 `textual_inversion_weight`）：
```python
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/training/sd_xl_base_finetune_textual_inversion.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0_ms.ckpt",
        "textual_inversion_weight": "runs/chinese_art/SD-XL-base-1.0_2000_ti.ckpt",  # or path to another textual inversion weight
    },
```
然后在 `demo/sampling.py` 的 `__main__` 中指定提示为 "a dog in <chinese-art> style" 并运行：
  ```shell
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
  streamlit run demo/sampling.py --server.port <your_port>
  ```

### 目标推理结果

使用提示 "a <cat-toy> backpack" 生成的图片如下所示：

<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/textual_inversion/sd-xl-a-cat-toy-backpack.png" width=850 />
</p>
<p align="center">
  <em> Figure 4. The generated images. </em>
</p>


### Style推理结果

使用提示 "a dog in <chinese-art> style" 生成的图片如下所示：

<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/textual_inversion/sd-xl-a-dog-in-chinese-art-style.png" width=850 />
</p>
<p align="center">
  <em> Figure 5. The generated images. </em>
</p>

# 参考

[1] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, Daniel Cohen-Or: An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. ICLR 2023
