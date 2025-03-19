
<hr>

<div align="center">
<h1>🚀 Janus-Series: Unified Multimodal Understanding and Generation Models</h1>

</div>


<div align="center">

  <a href="LICENSE-CODE">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53">
  </a>
  <a href="LICENSE-MODEL">
    <img alt="Model License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53">
  </a>
</div>


<p align="center">
  <a href="#2-model-download"><b>📥 Model Download</b></a> |
  <a href="#3-quick-start"><b>⚡ Quick Start</b></a> |
  <a href="#4-license"><b>📜 License</b></a> |
  <a href="#5-citation"><b>📖 Citation</b></a> <br>
  <!-- 📄 Paper Link (<a href="https://arxiv.org/abs/2410.13848"><b>Janus</b></a>, <a href="https://arxiv.org/abs/2410.13848"><b>JanusFlow</b></a>) | -->
  <!-- 🤗 Online Demo (<a href="https://huggingface.co/spaces/deepseek-ai/Janus-Pro-7B"><b>Janus-Pro-7B</b></a>, <a href="https://huggingface.co/spaces/deepseek-ai/Janus-1.3B"><b>Janus</b></a>, <a href="https://huggingface.co/spaces/deepseek-ai/JanusFlow-1.3B"><b>JanusFlow</b></a>) -->
</p>

We provide an efficient MindSpore implementation of [JanusPro](https://github.com/deepseek-ai/Janus). This repository is built on the models and code released by DeepSeek. We are grateful for their exceptional work and generous contribution to open source.


## News

**2025.03.12**: We have reproduced the multi-modal training pipelines referring to the JanusPro [paper](https://github.com/deepseek-ai/Janus), see [docs/training.md](docs/training.md).

**2025.02.10**: MindSpore implementation of Janus-Pro is released, supporting both multimodal understanding and visual generation on Ascend NPU.


## 1. Introduction

<a href="./janus_pro_tech_report.pdf"><b>Janus-Pro: Unified Multimodal Understanding and
Generation with Data and Model Scaling</b></a>

**Janus-Pro** is an advanced version of the previous work Janus. Specifically, Janus-Pro incorporates (1) an optimized training strategy, (2) expanded training data, and (3) scaling to larger model size. With these improvements, Janus-Pro achieves significant advancements in both multimodal understanding and text-to-image instruction-following capabilities, while also enhancing the stability of text-to-image generation.

<div align="center">
<img alt="image" src="https://github.com/user-attachments/assets/b1ca9876-08a5-4833-bc9a-2ba771269886" style="width:90%;">
</div>


## 2. Model Download

JanusPro is available to the public to support a broader and more diverse range of research within both academic and commercial communities.
Please note that the use of this model is subject to the terms outlined in [License section](#5-license). Commercial usage is
permitted under these terms.

### Huggingface

| Model                 | Sequence Length | Download                                                                    |
|-----------------------|-----------------|-----------------------------------------------------------------------------|
| Janus-Pro-1B | 4096            | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/Janus-Pro-1B) |
| Janus-Pro-7B | 4096        | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/Janus-Pro-7B) |


You can download by:

```shell
# with revision .safetensors can be downloaded
huggingface-cli download deepseek-ai/Janus-Pro-1B --revision refs/pr/6 --local-dir ckpts/Janus-Pro-1B
huggingface-cli download deepseek-ai/Janus-Pro-7B --revision refs/pr/110 --local-dir ckpts/Janus-Pro-7B
```



## 3. Quick Start

### Requirments

The code is tested in the following environments

| mindspore | ascend driver | firmware | cann tookit/kernel |
| :---:     |   :---:       | :---:    | :---:              |
| 2.5.0     |  24.1.0     |7.35.23    |   8.0.RC3.beta1   |


### Installation

On the basis of `Python >= 3.8` environment, install the necessary dependencies by running the following command:

```shell
pip install -e .
```


### Simple Inference Example

#### Multimodal Understanding

```shell
python inference.py \
    --image images/doge.png  \
    --question "explain this meme"
```

#### Text-to-Image Generation

```python
python generation_inference.py \
    --prompt "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair"

```

### Gradio Demo


For the remote gradio demo, you can run with the following command:


On NPU server:
```shell
pip install -e .[gradio]

python demo/app_januspro.py
```

On local terminal, run `ssh -L 37906:localhost:37906 user_name@server_ip`, then open `localhost:37906` on the web.

Have Fun!


### Training

Please refer to [training.md](docs/training.md)


## 4. Performance

### Multimodal Understanding

Experiments are tested on ascend 910* with mindspore 2.5.0 **graph** mode:

| model | # card(s) | image size | attn. type | throughput (token/s)|
|:-:|:-:|:-:|:-:|:-:|
| Janus-Pro-1B | 1 | 384x384 | Eager | 16.6|
| Janus-Pro-7B | 1 | 384x384 | Eager | 12.2|


Experiments are tested on ascend 910* with mindspore 2.5.0 **pynative** mode:

| model | # card(s) | image size | attn. type | throughput (token/s)|
|:-:|:-:|:-:|:-:|:-:|
| Janus-Pro-1B | 1 | 384x384 | Eager | 5.88 |
| Janus-Pro-7B | 1 | 384x384 | Eager | 3.30|

### Visual Generation

Experiments are tested on ascend 910* with mindspore 2.5.0 **graph** mode:

| model | # card(s) | batch Size | image size | attn. type | throughput (token/s)| s/img |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Janus-Pro-1B | 1 | 1 | 384x384 | Eager | 16.2 | ~ 40 |
| Janus-Pro-7B | 1 | 1 | 384x384 | Eager | 11.9 | ~ 52 |

Experiments are tested on ascend 910* with mindspore 2.5.0 **pynative** mode:

| model | # card(s) | batch size| image size | attn. type | throughput (token/s)| s/img |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Janus-Pro-1B | 1 | 1 | 384x384 | Eager | 4.52 | ~ 127|
| Janus-Pro-7B | 1 | 1 | 384x384 | Eager | 3.56 | ~ 162|

* All the performances are tested with KV-Cache enabled.

## 5. License

This code repository is licensed under [the MIT License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/HEAD/LICENSE-CODE). The use of Janus models is subject to [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/HEAD/LICENSE-MODEL).

## 6. Citation

```bibtex
@article{chen2025janus,
  title={Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling},
  author={Chen, Xiaokang and Wu, Zhiyu and Liu, Xingchao and Pan, Zizheng and Liu, Wen and Xie, Zhenda and Yu, Xingkai and Ruan, Chong},
  journal={arXiv preprint arXiv:2501.17811},
  year={2025}
}

```
