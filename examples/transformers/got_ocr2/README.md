# GOT-OCR 2.0: Advanced OCR Model for Text Extraction

[GOT-OCR 2.0](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) is an advanced Optical Character Recognition (OCR) model developed by StepFun that can accurately extract text from images. It supports multiple languages and various text layouts, making it suitable for document processing, receipt scanning, and general text extraction tasks.

## Introduction

GOT-OCR 2.0 is designed to handle complex OCR scenarios including:
- Multi-language text recognition
- Various text layouts and orientations
- Different font styles and sizes
- Noisy or degraded images
- Handwritten text recognition

The model uses advanced vision-language understanding to provide accurate text extraction with contextual awareness.

## Get Started

## 📦 Requirements
mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |




## Quick Start

### Basic OCR Usage

```python
import time
from transformers import AutoProcessor
import mindspore as ms
from mindone.transformers import AutoModelForImageTextToText

# Model configuration
MODEL_HUB = "stepfun-ai/GOT-OCR-2.0-hf"
IMAGE_PATH = "path/to/your/document.png"

# Load processor
start = time.time()
processor = AutoProcessor.from_pretrained(MODEL_HUB)
print(f"Loaded processor in {time.time()-start:.4f}s")

# Load model
start = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_HUB,
    mindspore_dtype=ms.bfloat16,
    attn_implementation="eager",
)
print(f"Loaded model in {time.time()-start:.4f}s")

# Process image
np_inputs = processor(IMAGE_PATH, return_tensors="np")
inputs = {}
for k, v in np_inputs.items():
    t = ms.Tensor(v)
    t = t.astype(ms.int32) if t.dtype == ms.int64 else t.astype(model.dtype)
    inputs[k] = t

# Generate OCR results
start = time.time()
output = model.generate(**inputs, max_new_tokens=4096)
print(f"OCR completed in {time.time()-start:.4f}s")

# Decode results
generated_ids = output[0][inputs["input_ids"].shape[1]:]
decoded_text = processor.decode(generated_ids, skip_special_tokens=True)
print("Extracted text:")
print(decoded_text)
```

### Run the Example

```bash
# Make sure you have an image file named demo.png in the current directory
python generate.py
```

## Features

- **Multi-language Support**: Recognizes text in multiple languages including English, Chinese, Japanese, Korean, etc.
- **Layout Understanding**: Handles various text layouts including multi-column documents, tables, and complex formatting
- **High Accuracy**: Advanced model architecture for accurate text recognition
- **Batch Processing**: Can process multiple images efficiently
- **Context Awareness**: Uses contextual understanding to improve recognition accuracy

## Model Architecture

GOT-OCR 2.0 combines:
- Vision encoder for image understanding and text region detection
- Text decoder for generating accurate text transcriptions
- Multimodal fusion for context-aware text extraction

## Performance

GOT-OCR 2.0 achieves state-of-the-art performance on various OCR benchmarks:

- **Text Recognition Accuracy**: High accuracy on standard OCR datasets
- **Multi-language Support**: Strong performance across different languages
- **Layout Preservation**: Maintains text structure and formatting
- **Speed**: Efficient processing for real-time applications

## Use Cases

- **Document Processing**: Extract text from PDFs, scanned documents, contracts
- **Receipt Scanning**: OCR for expense management and receipt processing
- **Form Processing**: Extract data from structured forms and applications
- **ID Card Recognition**: Extract information from identification documents
- **Book Digitization**: Convert physical books to digital text format
- **Accessibility**: Help visually impaired users access printed materials</contents>
</xai:function_call
<xai:function_call name="write">
<parameter name="file_path">/Users/weizheng/work/tmp/mindone/examples/transformers/helium/README.md
