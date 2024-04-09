# Text-to-Text Transfer Transformer

## 1. Introduction

The T5 (Text-To-Text Transfer Transformer) model is a series of Transformer-based language models developed by Google. It can be used for a wide range of NLP tasks, such as language translation, text classification, and question-answering. The authors convert these diverse tasks into a text-to-text format, and pretrain this model on a multi-task mixture both unsupervisedly and self-supervisedly. It has achieved state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more.


<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/t5/T5-diagram.PNG" width=550 />
</p>
<p align="center">
  <em> Figure 1. The Diagram of the T5 model. [<a href="#references">1</a>] </em>
</p>

The T5 model has different sizes, which are listed below:

- [google-t5/t5-small](https://huggingface.co/google-t5/t5-small)
- [google-t5/t5-base](https://huggingface.co/google-t5/t5-base)
- [google-t5/t5-large](https://huggingface.co/google-t5/t5-large)
- [google-t5/t5-3b](https://huggingface.co/google-t5/t5-3b)
- [google-t5/t5-11b](https://huggingface.co/google-t5/t5-11b)

## 2. Get Started
In this tutorial, we will introduce how to run inference with T5 model.

This tutorial includes:
- [x] Pretrained checkpoints conversion;
- [x] T5 text encoder;
- [ ] T5 text decoder.

### 2.1 Environment Setup


Please make sure the following frameworks are installed.

- python >= 3.7
- mindspore >= 2.2.10  [[install](https://www.mindspore.cn/install)]
- transformers >= 4.25.1 [[install](https://github.com/huggingface/transformers)] (for tokenizer)

Please install the other requirements using:
```bash
pip install -r requirement.txt
```
### 2.2 Pretrained Checkpoints

Taking `t5-v1_1-xxl` model as an example, please download the cache folder of the `t5-v1_1-xxl` model from HuggingFace [URL](https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main), and place it under `models/`. The t5 cache folder looks like:

```bash
models/t5-v1_1-xxl/
├── config.json
├── pytorch_model-00001-of-00002.bin
├── pytorch_model-00002-of-00002.bin
├── pytorch_model.bin.index.json
├── special_tokens_map.json
├── spiece.model
└── tokenizer_config.json
```

Then, you can convert the T5 Torch checkpoint to a MindSpore checkpoint using:

```bash
python tools/t5_converter.py --source models/t5-v1_1-xxl/pytorch_model-00001-of-00002.bin  models/t5-v1_1-xxl/pytorch_model-00002-of-00002.bin --target models/t5-v1_1-xxl/model.ckpt
```
This will save the converted MindSpore checkpoint file to `models/t5-v1_1-xxl/model.ckpt`.

## 3. Text Encoder

To extract the text embedding and the mask using the T5 text encoder, you can use:

```python
import os, sys
import mindspore as ms
ms.set_context(mode=0) # 0: graph mode; 1: pynative mode
mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
from mindone.models.t5 import T5Embedder

ckpt_path = "models/t5-v1_1-xxl/"
text_encoder = T5Embedder(cache_dir=ckpt_path, pretrained_ckpt=os.path.join(ckpt_path, "model.ckpt"))
text_emb, mask = text_encoder.get_text_embeddings("a red ball rolling on the ground.")
print(text_emb, mask)
```


## 4. Text Decoder

TBC


# References

[1] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. J. Mach. Learn. Res. 21: 140:1-140:67 (2020)
