# Qwen-Image

This repository provides the LoRA finetune codes of [Qwen-Image](https://arxiv.org/abs/2508.02324).

-----

## ✨ Key Features

* **Superior Text Rendering:** Qwen-Image excels at complex text rendering, including multiline layouts, paragraph-level semantics, and fine-grained details. It supports both alphabetic languages (e.g., English) and logographic languages (e.g., Chinese) with high fidelity.

* **Consistent Image Editing:** Through our enhanced multi-task training paradigm, Qwen-Image achieves exceptional performance in preserving both semantic meaning and visual realism during editing operations.

* **Strong Cross-Benchmark Performance:** Evaluated on multiple benchmarks, Qwen-Image consistently outperforms existing models across diverse generation and editing tasks, establishing a strong foundation model for image generation.



## 📑 Todo List
- Qwen-Image (Text-to-Image Model)
  - [x] LoRA finetune


## 🚀 Quick Start

### Requirements
| mindspore |	ascend driver | firmware    | cann toolkit/kernel|
| :-------: | :-----------: | :---------: | :----------------: |
| 2.7.0     |  25.2.0       | 7.7.0.6.236 | 8.2.RC1            |

### Installation
Clone the repo:
```sh
git clone https://github.com/mindspore-lab/mindone.git
cd mindone/examples/diffusers/qwenimage
```

Download Model Weights:
```bash
# Download from HuggingFace
hf download Qwen/Qwen-Image
```

### Run Qwen-Image LoRA FineTune

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
NPUS=2
MASTER_PORT=9000
LOG_DIR=outputs/lora
msrun --bind_core=True --worker_num=${NPUS} --local_worker_num=${NPUS} --master_port=${MASTER_PORT} --log_dir=${LOG_DIR}/parallel_logs \
python finetune_lora_with_mindspore_trainer.py \
    --output_dir ${LOG_DIR} \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --save_strategy no \
    --bf16
```
