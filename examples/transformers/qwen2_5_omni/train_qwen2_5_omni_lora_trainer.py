
"""
Qwen2.5-Omni model fine-tuning script using LoRA.

This script with default values fine-tunes a pretrained Thinker model from  Qwen2.5-Omni-3B/Qwen2.5-Omni-7B,
on the `linxy/LaTex_OCR` dataset ,

reference lora config: https://github.com/modelscope/ms-swift/pull/3613
"""


from dataclasses import dataclass, field

from PIL import Image
from tqdm.auto import tqdm
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser

import mindspore as ms

from mindone.transformers.mindspore_adapter import MindSporeArguments, init_environment
from mindone.transformers.models.llama import LlamaForSequenceClassification
from mindone.transformers.trainer import Trainer
from mindone.transformers.training_args import TrainingArguments

from mindone.transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniTalkerForConditionalGeneration,
)
from mindone.transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor

from mindspore.dataset import GeneratorDataset, transforms, vision
from mindone.diffusers._peft import LoraConfig
from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer
from mindone.diffusers._peft.utils import get_peft_model_state_dict, set_peft_model_state_dict


logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

class Qwen2_5OmniDataset():
    """
      Image VQA dataset for Qwen2.5-Omni model fine-tuning.
    """
    def __init__(self, dataset_path):
        dataset = load_dataset(dataset_path, name="human_handwrite")
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))

    def __getitem__(self, idx):



@dataclass
class MyArguments(MindSporeArguments, TrainingArguments):
    model_path: str = field(default="Qwen/Qwen2.5-Omni-3B")
    dataset_path: str = field(default="linxy/LaTex_OCR")
    output_dir: str = field(default="./outputs")
    enable_flash_attention: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    is_distribute: bool = field(default=False)
    lora_rank: int = field(default=16, metadata={"help": "The dimension of the LoRA update matrices."})
    # tune_mm_llm: bool = field(default=False)
    # tune_mm_mlp: bool = field(default=False)
    # tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    # video_max_frames: Optional[int] = field(default=8)
    # video_min_frames: Optional[int] = field(default=4)
    # data_flatten: bool = field(default=False)
    # data_packing: bool = field(default=False)
    # base_interval: int = field(default=2)
    # max_pixels: int = field(default=28 * 28 * 576)
    # min_pixels: int = field(default=28 * 28 * 16)
    # video_max_frame_pixels: int = field(default=32 * 28 * 28)
    # video_min_frame_pixels: int = field(default=4 * 28 * 28)
    system_prompt: str = field(default="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.")
    prompt: str = field(default="Please convert the image content into LaTex")


def freeze_params(m: nn.Cell):
    for p in m.get_parameters():
        p.requires_grad = False

def main():
    parser = HfArgumentParser(MyArguments, DataArguments)
    args, data_args = parser.parse_args_into_dataclasses()

    init_environment(args)

    # 1. Load the dataset
    dataset = load_dataset(args.dataset_path, name="human_handwrite")
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))

    system_prompt = args.system_prompt
    prompt = args.prompt
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    def process_function(examples):
        answer = examples["text"]
        image = examples["image"]
        conversations = [
            {'role': 'system', 'content': data_args.system_prompt},
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': data_args.prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
        ]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="np",
            padding=True,
        )

        # Prepare the labels, keep response part as labels
        prompt_ids = processor.tokenizer.apply_chat_template(conversations[:2], return_tensors="np")
        labels = np.ones_like(inputs["input_ids"]) * IGNORE_INDEX
        labels[..., len(prompt_ids) :] = inputs["input_ids"][..., len(prompt_ids) :]

        for k, v in inputs.items():
            examples[k] = v
        examples["labels"] = labels

        return examples

    tokenized_datasets = dataset.map(process_function, batched=True)
    small_train_dataset = tokenized_datasets["train"]
    small_eval_dataset = tokenized_datasets["test"]

    # 2. Load the model
    parent_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2" if args.enable_flash_attention else "eager",
        mindspore_dtype=ms.bfloat16 if args.bf16 else (ms.float16 if args.fp16 else None),
    ) # TODO: only load thinker state dicts
    model = parent_model.thinker
    model.config.use_cache = False
    freeze_params(model)
    # 2.2. Prepare the LoRA config
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=[
            "q",
            "k",
            "v",
            "proj",
            "k_proj",
            "v_proj",
            "q_proj",
            "out_proj",
            "o_proj",
        ],
    )
    model = get_peft_model(model, transformer_lora_config)
    if args.dtype == "fp16" or args.dtype == "bf16":
        cast_training_params(model, dtype=ms.float32)
    model.print_trainable_parameters()

    # 3. Prepare the evalutaion metric
    if args.do_eval:
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

    else:
        compute_metrics = None

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
