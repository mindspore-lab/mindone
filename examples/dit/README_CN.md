# 可扩展的基于转换器的扩散模型（DiT）

## 引言

以往常见的扩散模型（例如：稳定扩散模型）使用的是U-Net骨干网络，这缺乏可扩展性。DiT是一类基于转换器架构的新型扩散模型。作者设计了扩散转换器（DiTs），它们遵循视觉转换器（ViTs）[1]的最佳实践。它通过"patchify"将视觉输入作为一系列视觉标记的序列，然后由一系列转换器块（DiT块）处理这些输入。DiT模型和DiT块的结构如下所示：

<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/DiT_structure.PNG" width=550 />
</p>
<p align="center">
  <em> 图 1. DiT和DiT块的结构。 [<a href="#references">2</a>] </em>
</p>


DiTs是扩散模型的可扩展架构。作者发现网络复杂性（以Gflops计）与样本质量（以FID计）之间存在强相关性。换句话说，DiT模型越复杂，其在图像生成上的表现就越好。

## 开始使用

本教程将介绍如何使用MindONE运行推理和微调实验。

### 配套版本

| mindspore | ascend driver | firmware     | cann toolkit/kernel|
|:----------|:---           | :--          |:--|
| 2.3.1     | 24.1.RC2      | 7.3.0.1.231  | 8.0.RC2.beta1|

### 环境设置

```
pip install -r requirements.txt
```

### 预训练ckpt

我们参考[DiT的官方仓库](https://github.com/facebookresearch/DiT)下载预训练ckpt。目前，只有两个ckpt`DiT-XL-2-256x256`和`DiT-XL-2-512x512`可用。

下载`DiT-XL-2-{}x{}.pt`文件后，请将其放置在`models/`文件夹下，然后运行`tools/dit_converter.py`。例如，要转换`models/DiT-XL-2-256x256.pt`，您可以运行以下命令：
```bash
python tools/dit_converter.py --source models/DiT-XL-2-256x256.pt --target models/DiT-XL-2-256x256.ckpt
```

此外，还请从[huggingface/stabilityai.co](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)下载VAE ckpt，并通过运行以下命令进行转换：
```bash
python tools/vae_converter.py --source path/to/vae/ckpt --target models/sd-vae-ft-mse.ckpt
```

转换后，在`models/`下的ckpt应如下所示：
```bash
models/
├── DiT-XL-2-256x256.ckpt
├── DiT-XL-2-512x512.ckpt
└── sd-vae-ft-mse.ckpt
```

## 采样
要在Ascend设备上运行`DiT-XL/2`模型的`256x256`图像尺寸的推理，您可以使用：
```bash
python sample.py -c configs/inference/dit-xl-2-256x256.yaml
```

要在Ascend设备上运行`DiT-XL/2`模型的`512x512`图像尺寸的推理，您可以使用：
```bash
python sample.py -c configs/inference/dit-xl-2-512x512.yaml
```

要在GPU设备上运行相同的推理，只需按上述命令额外设置`--device_target GPU`。

默认情况下，我们以混合精度模式运行DiT推理，其中`amp_level="O2"`。如果您想以全精度模式运行推理，请在推理yaml文件中设置`use_fp16: False`。

对于扩散采样，我们使用与[DiT的官方仓库](https://github.com/facebookresearch/DiT)相同的设置：

- 默认采样器是DDPM采样器，默认采样步数是250。
- 对于无分类器引导，默认为引导比例是 $4.0$.

如果您想使用DDIM采样器并采样50步，您可以按以下方式修改推理yaml文件：
```yaml
# 采样
sampling_steps: 50
guidance_scale: 4.0
seed: 42
ddim_sampling: True
```

实验在MindSpore 2.3.1（图模式）的Ascend 910*上进行验证：

| model name | cards | image size | method | steps | jit level | ckpt loading time | graph compile | sample time |
| :--------: | :---: | :--------: | :----: | :---: | :-------: | :---------------: | :-----------: | :---------: |
|    dit     |   1   |  256x256   |  ddpm  |  250  |    O2     |      16.41s       |    82.83s     |   58.45s    |

一些生成的示例图像如下所示：
Some generated example images are shown below:
<p float="center">
<img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/dit/512x512/class-207.png" width="25%" /><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/dit/512x512/class-360.png" width="25%" /><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/dit/512x512/class-417.png" width="25%" /><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/dit/512x512/class-979.png" width="25%" />
</p>
<p float="center">
<img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/dit/256x256/class-207.png" width="12.5%" /><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/dit/256x256/class-279.png" width="12.5%" /><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/dit/256x256/class-360.png" width="12.5%" /><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/dit/256x256/class-387.png" width="12.5%" /><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/dit/256x256/class-417.png" width="12.5%" /><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/dit/256x256/class-88.png" width="12.5%" /><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/dit/256x256/class-974.png" width="12.5%" /><img src="https://raw.githubusercontent.com/jianyunchao/mindone-assets/v0.2.0/dit/256x256/class-979.png" width="12.5%" />
</p>

## 使用ImageNet数据集进行模型训练

对于`mindspore>=2.3.0`，建议使用msrun启动使用ImageNet数据集格式的分布式训练，使用命令如下：
```bash
msrun --worker_num=4 \
    --local_worker_num=4 \
    --bind_core=True \
    --log_dir=msrun_log \
    python train.py \
    -c configs/training/class_cond_train.yaml \
    --data_path PATH_TO_YOUR_DATASET \
    --use_parallel True
```

您可以使用以下命令使用ImageNet数据集格式启动分布式训练：
```bash
mpirun -n 4 python train.py \
    -c configs/training/class_cond_train.yaml \
    --data_path PATH_TO_YOUR_DATASET \
    --use_parallel True
```

其中`PATH_TO_YOUR_DATASET`是您的ImageNet数据集的路径，例如`ImageNet2012/train`。

对于配备Ascend设备的机器，您也可以使用rank table启动分布式训练。
请运行：
```bash
bash scripts/run_distributed.sh path_of_the_rank_table 0 4 path_to_your_dataset
```

启动4P训练。有关训练脚本的详细用法，请运行：
```bash
bash scripts/run_distributed.sh -h
```

## 评估

实验在MindSpore 2.3.1（图模式）的Ascend 910*上进行验证：

| model name | cards | image size | graph compile | batch size | recompute | dataset sink mode | jit level | step time | train. imgs/s |
| :--------: | :---: | :--------: | :-----------: | :--------: | :-------: | :---------------: | :-------: | :-------: | :-----------: |
|    dit     |   1   |  256x256   |   3~5 mins    |     64     |    OFF    |        ON         |    O2     |   0.89s   |     71.91     |
|    dit     |   1   |  256x256   |   3~5 mins    |     64     |    ON     |        ON         |    O2     |   0.95s   |     67.37     |
|    dit     |   4   |  256x256   |   3~5 mins    |     64     |    ON     |        ON         |    O2     |   1.03s   |    248.52     |
|    dit     |   8   |  256x256   |   3~5 mins    |     64     |    ON     |        ON         |    O2     |   0.93s   |    515.61     |


# 参考文献

[1] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2020. 1, 2, 4, 5

[2] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023
