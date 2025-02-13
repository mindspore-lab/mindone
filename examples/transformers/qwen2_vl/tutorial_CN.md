# 教程：从零开始实现 Qwen2-VL (MindSpore版本)
中文教程  &nbsp;&nbsp;|&nbsp;&nbsp; [English Tutorial](tutorial.md)

> 基于 [MindSpore](https://gitee.com/mindspore/mindspore) and [MindONE](https://github.com/mindspore-lab/mindone) 实现Qwen2-VL。

**简介：** Qwen2-VL是大规模视觉语言模型[Qwen-VL](https://github.com/QwenLM/Qwen-VL)的升级版本。和Qwen-VL相比，Qwen2-VL增强了图片压缩，升级了视频理解，可以进一步整合视觉智能体(agent)的功能，以及支持多国语言的交互。 
<br>
优化的Qwen2-VL模型通过[Naive Dynamic Resolution](#naive-dynamic-resolution)来处理任意的图像分辨率，并且利用[Multimodal Rotary Position Embedding (M-ROPE)](#multimodal-rotary-position-embedding-m-rope)来高效同时处理一维文字和多位视觉数据。升级的Qwen2-VL模型在视觉任务上展示了与领先的智能系统诸如GPT-4o和Claude 3.5 Sonnet具有竞争力的表现， 并且在一众开源模型中有较高的文字处理和理解能力。这使得Qwen2-VL具有多模态的处理和理解能力，从而成为了服务于多种应用的多功能工具。

**开发环境需求:** Python, Mindspore, Mindone, Transformers (最新版>=v.4.52)

## 1. 流程概览
Qwen2-VL 引用 [ViT](https://github.com/google-research/vision_transformer#vision-transformer)的编码器用作视觉编码器(Vision Encoder), 和大语言模型(LLM) [Qwen2](https://github.com/QwenLM/Qwen2)的解码器作为模型的解码器(Decoder).

### 整体流程图

<div style="display: block; margin-left: auto;  margin-right: auto; width:80%" >
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/qwen2_vl.jpg" alt="Qwen2-VL Framework" width="100%" />

_Qwen2-VL 模型架构. 图源: [原论文](https://arxiv.org/abs/2409.12191)._
<br>整体流程：Feeding multimodal inputs (vision and text) with M-ROPE into ViT visual encoder, LLM Qwen2 decode encoded input tokens and return textual reponses.

</div>