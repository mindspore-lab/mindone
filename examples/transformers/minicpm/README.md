# MiniCPM4

# Introduction
MiniCPM 4 is an extremely efficient large end-side model. It has been efficiently optimized from four aspects: model architecture, learning algorithm, training data and reasoning system, achieving an extreme efficiency improvement.

### 🏗️ Efficient model architecture:  
InfLLM v2 - Trainable Sparse Attention Mechanism: By adopting a trainable sparse attention mechanism architecture, in the processing of 128K long text, each lexical element only needs to perform correlation calculation with less than 5% of the lexical elements, significantly reducing the computational overhead of long text  
### 🧠 Efficient learning algorithm:  
Model Wind Tunnel 2.0 - Efficient Predictable Scaling: Introducing a Scaling prediction method for downstream tasks to achieve more accurate model training configuration search  
BitCPM - Ultimate Three-Value Quantization: Compressing the model parameter bit width to 3 values, achieving an ultimate 90% reduction in the model bit width  
Efficient training engineering optimization: Adopt FP8 low-precision computing technology and combine it with the Multi-token Prediction training strategy  
### 📚 High knowledge density training data:  
UltraClean - Cleaning and Synthesis of High-Quality Pre-trained Data: Build an iterative data cleaning strategy based on efficient validation, and open source the high-quality Chinese and English pre-trained dataset UltraFineweb  
UltraChat v2 - High-quality Supervised Fine-tuning Data Synthesis: Build large-scale high-quality supervised fine-tuning datasets, covering multiple dimensions such as knowledge-intensive data, inference-intensive data, instruction compliance data, long text understanding data, and tool call data  
### ⚡ Efficient reasoning system:  
CPM.cu - A lightweight and efficient CUDA inference framework: Integrating sparse attention mechanism, model quantization and speculative sampling, it fully demonstrates the efficiency advantages of MiniCPM4  
ArkInfer - Cross-platform Deployment System: Supports one-click deployment in multiple backend environments and provides flexible cross-platform adaptation capabilities

# Get Started

## 📦 Requirements
mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |



git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install mindone==0.4.0
cd examples/transformers/minicpm
```

```shell
# If using paged attention on mindspore 2.7.0, one environment variable need to be set for stopping tracing parameter memory

export MS_ENABLE_TRACE_MEMORY=OFF
```

## Quick Start

Here is a usage example of MiniCPM-8B. you can use the following command:

```bash
# for MiniCPM4-8B inference
python generate.py
```

In addition, model.chat interface has been supported for MiniCPM-0.5B interface.
```bash
# for MiniCPM4-0.5B inference
python chat.py
```

## Inference Speed
|      model name	      | mindspore version | precision* | cards | page attn | 	tokens/s	 |
|:---------------------:|:-----------------:|:----------:|:---:  | :---:  |:----------:|
| openbmb/MiniCPM4-0.5B |       2.6.0       |    bf16    | 1 | ✅  |    11.9    |
|  openbmb/MiniCPM4-8B  |       2.6.0       |    bf16    |     1      | ✅  |    9.6     |
| openbmb/MiniCPM4-0.5B |       2.7.0       |    bf16    | 1 | ✅  |    15.3    |
|  openbmb/MiniCPM4-8B  |       2.7.0       |    bf16    |     1      | ✅  |   13.38    |
