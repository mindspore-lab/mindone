# HerBERT: Polish Language Model

[HerBERT](https://huggingface.co/allegro/herbert-klej-cased-v1) is a Polish language model developed by Allegro, based on the RoBERTa architecture. It's specifically trained on Polish language data and optimized for various Polish NLP tasks including text classification, named entity recognition, and question answering.

## Introduction

HerBERT is designed to provide high-quality Polish language understanding and generation capabilities. It comes in different sizes (base, large) and is trained on a comprehensive Polish corpus, making it suitable for various Polish language processing tasks.

## Get Started

## 📦 Requirements
mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |




## Quick Start

### Basic Usage

```python
from transformers import HerbertTokenizer
from mindspore import Tensor
from mindone.transformers import RobertaModel

# Available model variants
model_names = {
    "herbert-klej-cased-v1": {
        "tokenizer": "allegro/herbert-klej-cased-tokenizer-v1",
        "model": "allegro/herbert-klej-cased-v1",
        "ref": "refs/pr/1",
    },
    "herbert-base-cased": {
        "tokenizer": "allegro/herbert-base-cased",
        "model": "allegro/herbert-base-cased",
        "ref": "refs/pr/2",
    },
    "herbert-large-cased": {
        "tokenizer": "allegro/herbert-large-cased",
        "model": "allegro/herbert-large-cased",
        "ref": "refs/pr/2",
    },
}

# Load model and tokenizer
model_name = "herbert-klej-cased-v1"
tokenizer = HerbertTokenizer.from_pretrained(model_names[model_name]["tokenizer"])
model = RobertaModel.from_pretrained(
    model_names[model_name]["model"],
    revision=model_names[model_name]["ref"]
)

# Example Polish text
text = "Kto ma lepszą sztukę, ma lepszy rząd – to jasne."

# Tokenize input
encoded_input = tokenizer.encode(text, return_tensors="np")
input_tensor = Tensor(encoded_input)

# Get model output
output = model(input_tensor)
last_hidden_states = output.last_hidden_state

print(f"Input text: {text}")
print(f"Tokenized: {encoded_input}")
print(f"Output shape: {last_hidden_states.shape}")
```

### Run the Example

```bash
python generate.py
```

## Model Variants

- **HerBERT-KLEJ-cased-v1**: Base model trained on KLEJ benchmark
- **HerBERT-base-cased**: Base model with 12 layers
- **HerBERT-large-cased**: Large model with 24 layers for better performance

## Features

- **Polish Language Focus**: Specifically trained and optimized for Polish language
- **Multiple Model Sizes**: Available in base and large configurations
- **Comprehensive Training**: Trained on diverse Polish language datasets
- **Task Agnostic**: Can be fine-tuned for various Polish NLP tasks

## Model Architecture

HerBERT is based on RoBERTa architecture with:
- **Bidirectional Attention**: Processes text in both directions simultaneously
- **Byte-level BPE Tokenization**: Efficient tokenization for Polish text
- **Large-scale Pre-training**: Trained on extensive Polish language corpora
- **Task-specific Fine-tuning**: Adaptable to various downstream tasks

## Performance

HerBERT achieves strong performance on Polish language tasks:

- **KLEJ Benchmark**: State-of-the-art results on Polish language understanding tasks
- **Named Entity Recognition**: High accuracy for Polish NER tasks
- **Text Classification**: Excellent performance on Polish text classification
- **Question Answering**: Strong results on Polish QA datasets

## Use Cases

- **Polish Text Classification**: Sentiment analysis, topic categorization
- **Named Entity Recognition**: Identifying Polish persons, organizations, locations
- **Question Answering**: Polish language QA systems
- **Text Summarization**: Summarizing Polish language content
- **Language Understanding**: General Polish language comprehension tasks
- **Content Moderation**: Filtering Polish language content
- **Chatbots**: Polish language conversational AI</contents>
</xai:function_call
<xai:function_call name="write">
<parameter name="file_path">/Users/weizheng/work/tmp/mindone/examples/transformers/kimi_vl/README.md
