[中文](./README_CN.md)|[English](./README.md)
# 盘古画画3.0

本例为盘古画画3.0模型的昇思MindSpore实现。


## 特性

盘古画画3.0在2.0基础上，在多语言、多尺寸、画质、模型放大等多个方面进行尝试和更新，其中包括：

- [x] 网络结构扩容，参数量从1B扩大到5B，是当前**业界最大的中文文生图模型**；
- [x] 支持**中英文双语**输入；
- [x] 提升输出分辨率，支持**原生1K输出**（v1->v2->v3: 512->768->1024）；
- [x] 多尺寸（16:9、4:3、2:1...）输出；
- [x] **可量化的风格化调整**：动漫、艺术性、摄影控制；
- [x] 基于**昇腾硬件和昇思平台**进行大规模多机多卡训练、推理，全自研昇思MindSpore平台和昇腾Ascend硬件；
- [x] 采用**自研RLAIF**提升画质和艺术性表达。


## 进展

**2023年12月12日**

支持盘古画画3.0的文生图推理任务。


## 快速上手

### 安装

确保以下依赖已正常安装：

- python >= 3.7
- mindspore >= 2.2.10  [[点击安装](https://www.mindspore.cn/install)]

通过运行下述命令安装所需依赖：
```shell
pip install -r requirements.txt
```

### 预训练参数

盘古画画的文生图任务需要同时准备`low timestamp model`和`high timestamp model`两个模型的预训练参数（我们将在后续开源相关ckpt）


| **模型** | **MindSpore Checkpoint** |
|-------------|--------------------------|
| Pangu3-low-timestamp-model | [pangu_low_timestamp-127da122.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/PanGu-Draw-v3/pangu_low_timestamp-127da122.ckpt) |
| Pangu3-high-timestamp-model | [pangu_high_timestamp-c6344411.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/PanGu-Draw-v3/pangu_high_timestamp-c6344411.ckpt) |


### 推理

确认你已准备好相关预训练模型参数后，通过运行下述命令进行文生图推理任务：

```shell
# run txt2img on Ascend
export MS_PYNATIVE_GE=1
python demo/pangu/pangu_sampling.py \
--device_target "Ascend" \
--ms_amp_level "O2" \
--config "configs/inference/pangu_sd_xl_base.yaml" \
--high_solution \
--weight "path/to/low_timestamp_model.ckpt" \
--high_timestamp_weight "path/to/high_timestamp_model.ckpt" \
--prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
```


## 样例

注: 样例基于昇腾910*硬件，通过40步采样得到

<div align="center">
<img src="https://github.com/Stability-AI/generative-models/assets/143256262/0a8b0b1a-3b54-4a35-beec-4163be61fa88" width="320" />
<img src="https://github.com/Stability-AI/generative-models/assets/143256262/d14a3761-47ad-4671-adcd-866d9d6e6def" width="320" />
<img src="https://github.com/Stability-AI/generative-models/assets/143256262/42e18b39-5c62-4d0f-aa3b-8c0c6e0715f2" width="320" />
</div>
<p align="center">
<font size=3>
<em> 图1: "一幅中国水墨画：一叶轻舟漂泊在波光粼粼的湖面上，舟上的人正在饮酒放歌" </em> <br>
<em> 图2: "坐在海边看海浪的少年，黄昏" </em> <br>
<em> 图3: "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" </em> <br>
</font>
</p>
<br>
