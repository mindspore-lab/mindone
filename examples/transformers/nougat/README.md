# Nougat

## Overview

The Nougat model was proposed
in [Nougat: Neural Optical Understanding for Academic Documents](https://arxiv.org/abs/2308.13418) by Lukas Blecher,
Guillem Cucurull, Thomas Scialom, Robert Stojnic. Nougat uses the same architecture as Donut, meaning an image
Transformer encoder and an autoregressive text Transformer decoder to translate scientific PDFs to markdown, enabling
easier access to them.

The abstract from the paper is the following:

> Scientific knowledge is predominantly stored in books and scientific journals, often in the form of PDFs. However, the
> PDF format leads to a loss of semantic information, particularly for mathematical expressions. We propose Nougat (Neural
> Optical Understanding for Academic Documents), a Visual Transformer model that performs an Optical Character
> Recognition (OCR) task for processing scientific documents into a markup language, and demonstrate the effectiveness of
> our model on a new dataset of scientific documents. The proposed approach offers a promising solution to enhance the
> accessibility of scientific knowledge in the digital age, by bridging the gap between human-readable documents and
> machine-readable text. We release the models and code to accelerate future work on scientific text recognition.

> [!NOTE]
> The model is identical to Donut in terms of architecture.

## 📦 Requirements
mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |




## Inference

Nougat’s `VisionEncoderDecoder` model accepts images as input and makes use of `generate()` to autoregressively generate
text given the input image.

The `NougatImageProcessor` class is responsible for preprocessing the input image and `NougatTokenizerFast` decodes the
generated target tokens to the target string. The `NougatProcessor` wraps `NougatImageProcessor` and `NougatTokenizerFast`
classes into a single instance to both extract the input features and decode the predicted token ids.

### Example

```python
from datasets import load_dataset
from transformers import NougatProcessor

from mindspore import tensor

from mindone.transformers import VisionEncoderDecoderModel

processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

# prepare PDF image for the model
dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
image = dataset["test"][0]["image"].convert("RGB")
pixel_values = tensor(processor(image, return_tensors="np").pixel_values)

# generate transcription (here we only generate 30 tokens)
outputs = model.generate(
    pixel_values, min_length=1, max_new_tokens=30, bad_words_ids=[[processor.tokenizer.unk_token_id]]
)

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)
print(sequence)
```

**Output:**

```commandline
11:14 to
11:39 a.m.
Coffee Break
11:39 a.
```
