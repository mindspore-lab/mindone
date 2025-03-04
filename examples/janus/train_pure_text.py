import os
import sys

import urllib3
from datasets import load_dataset

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import numpy as np

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoConfig

from mindone.transformers.trainer import Trainer
from mindone.transformers.training_args import TrainingArguments

ms.set_context(mode=0)

dataset = load_dataset("pubmed_qa", "pqa_labeled")
system_prompt = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)
vl_chat_processor = VLChatProcessor = VLChatProcessor.from_pretrained("deepseek-ai/Janus-Pro-1B")
tokenizer = vl_chat_processor.tokenizer


def preprocess(examples, BatchInfo):
    for example in examples:
        question = example["question"]
        answer = example["long_answer"]

        conversations = [
            {
                "role": "<|User|>",
                "content": question,
            },
            {"role": "<|Assistant|>", "content": answer},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversations,
            system_prompt=system_prompt,
        )

        inputs = tokenizer(
            sft_format,
            padding="max_length",
            max_length=512,
            truncation=True,
        )

        answer_tokens = tokenizer(
            answer,
            truncation=True,
            add_special_tokens=True,
        )["input_ids"]

        input_ids = inputs["input_ids"]

        start_index = len(input_ids) - len(answer_tokens)
        end_index = len(input_ids) - 2

        ignore_index = -100
        labels = inputs["input_ids"]
        for i in range(len(labels)):
            if i < start_index and i > end_index:
                labels[i] = ignore_index

        inputs["input_ids"] = np.array(inputs["input_ids"])
        inputs["attention_mask"] = np.array(inputs["attention_mask"])
        inputs["labels"] = np.array(labels)

    yield {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels}


config = AutoConfig.from_pretrained("deepseek-ai/Janus-Pro-1B")
language_config = config.language_config
language_config._attn_implementation = "eager"
model = MultiModalityCausalLM.from_pretrained(
    "deepseek-ai/Janus-Pro-1B",
    language_config=language_config,
    trust_remote_code=True,
    revision="refs/pr/1",
)

model = model.language_model
model.set_train()

training_args = TrainingArguments(
    output_dir="results_graphs",
    per_device_train_batch_size=1,
    num_train_epochs=10,
    weight_decay=0.01,
    learning_rate=1e-5,
    logging_dir="./logs_graphs",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    data_collator=preprocess,
)

trainer.train()
