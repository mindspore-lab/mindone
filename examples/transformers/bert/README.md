# BERT: Bidirectional Encoder Representations from Transformers

[BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google. BERT is designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context.

## Introduction

BERT is pre-trained on a large corpus of text data using two unsupervised tasks:
- **Masked Language Modeling (MLM)**: Randomly masks some words in a sentence and trains the model to predict them
- **Next Sentence Prediction (NSP)**: Trains the model to understand relationships between sentences

This pre-training allows BERT to be fine-tuned for various downstream NLP tasks with minimal task-specific modifications.

## Get Started

## 📦 Requirements
mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |

## Quick Start

### Text Classification Fine-tuning

```python
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import mindspore as ms
from mindspore import nn
from mindone.transformers.mindspore_adapter import HF2MSDataset, TrainOneStepWrapper
from mindone.transformers.models.bert import BertForSequenceClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="google-bert/bert-base-cased",
                       help="pretrained model name")
    parser.add_argument("--dataset_path", type=str, default="Yelp/yelp_review_full",
                       help="dataset path.")
    args = parser.parse_args()

    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

    # 1. Load and prepare dataset
    dataset = load_dataset(args.dataset_path)
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # 2. Create model
    model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels=5)
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=2e-5)

    # 3. Create dataset
    train_dataset = HF2MSDataset(tokenized_datasets["train"])
    train_loader = ms.dataset.GeneratorDataset(
        train_dataset, column_names=list(train_dataset.column_names), shuffle=True
    ).batch(8)

    # 4. Training
    train_step = TrainOneStepWrapper(model, optimizer)
    train_step.set_train()

    for batch in train_loader:
        loss = train_step(**batch)
        print(f"Loss: {loss}")

if __name__ == "__main__":
    main()
```

### Run the Examples

```bash
# Fine-tune BERT on Yelp review classification (native MindSpore)
python finetune_in_native_mindspore.py --model_path google-bert/bert-base-cased --dataset_path Yelp/yelp_review_full

# Fine-tune BERT using MindSpore Trainer
python finetune_with_mindspore_trainer.py
```

## Model Variants

- **BERT-base**: 12-layer, 768-hidden, 12-heads, 110M parameters
- **BERT-large**: 24-layer, 1024-hidden, 16-heads, 340M parameters
- **BERT-base-multilingual**: Supports 104 languages
- **BERT-base-chinese**: Optimized for Chinese text

## Features

- **Bidirectional Context**: Considers both left and right context simultaneously
- **Deep Understanding**: Learns rich contextual representations
- **Task Agnostic**: Can be fine-tuned for various NLP tasks
- **Multilingual Support**: Available in multiple languages

## Applications

- **Text Classification**: Sentiment analysis, topic classification
- **Named Entity Recognition**: Identifying persons, organizations, locations
- **Question Answering**: Extractive QA on SQuAD dataset
- **Natural Language Inference**: Determining relationship between sentences
- **Feature Extraction**: Used as backbone for other NLP models

## Performance

BERT achieves state-of-the-art results on various NLP benchmarks including:
- GLUE benchmark (General Language Understanding Evaluation)
- SQuAD (Stanford Question Answering Dataset)
- MultiNLI (Multi-Genre Natural Language Inference)</contents>
</xai:function_call
<xai:function_call name="write">
<parameter name="file_path">/Users/weizheng/work/tmp/mindone/examples/transformers/got_ocr2/README.md
