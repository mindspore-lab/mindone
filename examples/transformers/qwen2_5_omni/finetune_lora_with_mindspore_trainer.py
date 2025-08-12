
"""
Qwen2.5-Omni model fine-tuning script using LoRA.

This script with default values fine-tunes a pretrained Thinker model from  Qwen2.5-Omni-3B/Qwen2.5-Omni-7B,
on the `linxy/LaTex_OCR` dataset ,

reference lora config: https://github.com/modelscope/ms-swift/pull/3613
reference data processing: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/qwenvl/data/data_qwen.py

Usage:
```
DEVICE_ID=0 python finetune_lora_with_mindspore_trainer.py \
    --model_path Qwen/Qwen2.5-Omni-3B \
    --lora_rank 8 \
    --lora_alpha 16 \
    --dataset_path linxy/LaTex_OCR \
    --output_dir ./outputs/lora \
    --num_train_epochs 1 \
    --eval_strategy no \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --save_steps 500 \
    --logging_steps 1 \
    --save_total_limit 1 \
    --download_num_workers 4
```
"""

from dataclasses import dataclass, field
import os
import math
from PIL import Image
from tqdm.auto import tqdm
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser
from typing import Union

import mindspore as ms
from mindspore import nn

from mindone.transformers.mindspore_adapter import MindSporeArguments, init_environment
from mindone.transformers.trainer import Trainer
from mindone.transformers.training_args import TrainingArguments

from mindone.transformers import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2Tokenizer
from transformers.models.qwen2_5_omni import Qwen2_5OmniConfig
from mindone.transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor

from mindspore.dataset import GeneratorDataset, transforms, vision
from mindone.diffusers._peft import LoraConfig, get_peft_model
from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer
from mindone.diffusers._peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from mindone.trainers import create_optimizer
from mindone.transformers.optimization import get_scheduler
from mindone.diffusers.training_utils import cast_training_params
from qwen_omni_utils import process_mm_info

import logging
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass
class MyArguments(MindSporeArguments, TrainingArguments):
    mode: int = field(default=ms.PYNATIVE_MODE, metadata={"help": "Graph(not supported)/Pynative"})
    model_path: str = field(default="Qwen/Qwen2.5-Omni-3B")
    dataset_path: str = field(default="linxy/LaTex_OCR")
    output_dir: str = field(default="./outputs")
    enable_flash_attention: bool = field(default=True)
    gradient_checkpointing: bool = field(default=False) # LoRA does not support
    is_distribute: bool = field(default=False)
    lora_rank: int = field(default=8, metadata={"help": "The dimension of the LoRA update matrices."})
    lora_alpha: int = field(default=16, metadata={"help": "The scaling factor alpha of the LoRA."})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Batch size per device for training."})
    resume: Union[bool, str] = field(default=False, metadata={"help": "Resume training from a checkpoint."})

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    max_length: int = field(default=4096, metadata={"help": "Fixed token length for training."})
    system_prompt: str = field(default="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.")
    prompt: str = field(default="Please convert the image content into LaTex")


def freeze_params(m: nn.Cell):
    for p in m.get_parameters():
        p.requires_grad = False

def main():
    parser = HfArgumentParser((MyArguments, DataArguments))
    args, data_args = parser.parse_args_into_dataclasses()

    init_environment(args)

    # 1. Load the dataset
    if os.path.isdir(args.dataset_path):
        dataset = load_dataset("parquet", args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path, name="human_handwrite")
    dataset["train"] = dataset["train"].shuffle(seed=42) # 1.2k
    dataset["test"] = dataset["test"]                    # 70

    system_prompt = data_args.system_prompt
    prompt = data_args.prompt
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    def process_function(examples):
        answer = examples["text"]
        image = examples["image"].convert("RGB")
        if args.per_device_train_batch_size > 1:
            image = image.resize((512, 128)) # use homogeneous size for training batch
        conversations = [
            {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
        ]
        input_padding = "max_length" if args.per_device_train_batch_size > 1 else True
        # for batched training, use homogeneous token length, padding side is left
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="np",
            # kwargs to be passed to `Qwen2_5OmniProcessor`
            padding=input_padding,
            max_length=data_args.max_length,
            use_audio_in_video=False,
        ) # input_ids, attention_mask, pixel_values, image_grid_thw

        # Prepare the labels, keep response part as labels
        if input_padding == "max_length":
            input_full_ids = processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
                return_tensors="np",
                padding=True,
            )[0] # only ids
        else:
            input_full_ids = inputs["input_ids"][0]
        prompt_ids = processor.apply_chat_template(
            conversations[:-1],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
            return_tensors="np",
            padding=True,
        )[0] # only ids
        labels = np.ones_like(inputs["input_ids"]) * IGNORE_INDEX
        response_start_id = inputs["input_ids"].shape[1] - len(input_full_ids) + len(prompt_ids)
        labels[..., response_start_id :] = inputs["input_ids"][..., response_start_id :]

        for k, v in inputs.items():
            if v.shape[0] == 1:
                examples[k] = v[0]  # remove batch dimension
            else:
                examples[k] = v # pixel_values
        examples["labels"] = labels[0]
        examples.pop("text")  # remove text from examples
        examples.pop("image")  # remove image from examples

        return examples

    tokenized_datasets = dataset.map(process_function, batched=False)
    small_train_dataset = tokenized_datasets["train"]
    small_eval_dataset = tokenized_datasets["test"]

    dataset_len = len(small_train_dataset)
    num_update_steps_per_epoch = max(1, dataset_len // args.gradient_accumulation_steps)
    num_training_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

    # 2. Load the model
    # 2.1. Load pretrained model
    parent_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2" if args.enable_flash_attention else "eager",
        mindspore_dtype=ms.bfloat16 if args.bf16 else (ms.float16 if args.fp16 else None),
    ) # TODO: only load thinker state dicts
    model = parent_model.thinker
    model.config.use_cache = False
    freeze_params(model)

    # 2.2. Prepare the LoRA config
    # all attn linear layers
    vision_enc_modules = ["q", "k", "v", "attn.proj"]
    # audio_enc_modules = ["k_proj", "v_proj", "q_proj", "out_proj", "o_proj"] # shared same names with text model
    audio_enc_modules = []
    qwen25omni_attn_modules = []
    for i in range(model.config.text_config.num_hidden_layers):
        qwen25omni_attn_modules.append(f"model.layers.{i}.self_attn.q_proj")
        qwen25omni_attn_modules.append(f"model.layers.{i}.self_attn.k_proj")
        qwen25omni_attn_modules.append(f"model.layers.{i}.self_attn.v_proj")
        qwen25omni_attn_modules.append(f"model.layers.{i}.self_attn.o_proj")
    target_modules = vision_enc_modules + audio_enc_modules + qwen25omni_attn_modules
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    model.is_gradient_checkpointing = False
    model = get_peft_model(model, lora_config)
    if args.fp16 or args.bf16:
        cast_training_params(model, dtype=ms.float32)
    model.print_trainable_parameters()
    # print(model.target_module_names)

    # 3. [optional] Prepare the evalutaion metric
    if args.do_eval: # TODO: do not support yet
        metric = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
    else:
        compute_metrics = None

    # 4. Training setups: lr scheduler, optimizer, trainer, etc.
    # lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        base_lr=args.learning_rate,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=args.lr_scheduler_kwargs,
    )
    # [required] optimizer
    # FIXME: since only train lora layer,
    # auto-creating optimizer in transformers Trainer may occur empty params list since there is not trainable layernorm layers.
    optimizer_kwargs = {
        "name": "adamw",
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
        "lr": lr_scheduler,
    }
    optimizer = create_optimizer(model.get_base_model().trainable_params(), **optimizer_kwargs)

    # trainer
    trainer = Trainer(
        model=model.get_base_model(), # use base model for parsing construct() arguments
        args=args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
        optimziers=(optimizer, lr_scheduler), # for LoRA
    )

    # trainer.train(resume_from_checkpoint=args.resume) # FIXME: do not support resume training yet
    # FIXME: now use the code below temorarily
    if isinstance(args.resume, str) or (isinstance(args.resume, bool) and args.resume):
        from transformers.trainer_utils import get_last_checkpoint
        from transformers.trainer_callback import TrainerState
        TRAINER_STATE_NAME = "trainer_state.json"
        resume_from_checkpoint = None
        resume_path = args.resume if isinstance(args.resume, str) else args.output_dir
        if resume_from_checkpoint is None:
            raise ValueError(f"No valid checkpoint found in {resume_path}.")
        trainer._load_from_checkpoint(resume_from_checkpoint)
        trainer.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        trainer.args.num_train_epochs -= trainer.state.epoch

    # train the model and save the LoRA weights
    if trainer.args.num_train_epochs > 0:
        trainer.train()
        model.save_pretrained(os.path.join(args.output_dir, "lora"))

    # 5. Inference and evaluation
    model.merge_and_unload()  # merge LoRA weights into the base model
    parant_model.thinker = model.get_base_model()  # replace thinker with LoRA-enhanced model
    parant_model.set_train(False)

    # inference function
    def inference(medium_path, prompt, medium_type="image", use_audio_in_video=False):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        medium = None
        if medium_type == "image":
            medium = {
                "type": medium_type,
                "image": medium_path,
                "max_pixels": 360 * 420,
            }
        if medium is not None:
            messages[1]["content"].append(medium)

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="np",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )

        # convert input to Tensor
        for key, value in inputs.items():  # by default input numpy array or list
            inputs[key] = ms.Tensor(value)
            if inputs[key].dtype == ms.int64:
                inputs[key] = inputs[key].to(ms.int32)
            else:
                inputs[key] = inputs[key].to(parent_model.dtype)

        text_ids = parent_model.generate(**inputs, use_audio_in_video=use_audio_in_video, return_audio=False)
        text_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, text_ids)]
        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text

    correct = 0
    file_path = os.path.join(args.output_dir, "inference_lora_eval.txt")
    for idx, example in enumerate(small_eval_dataset):
        medium = example["image"].convert("RGB")
        answer = example["text"]
        response = inference(medium, prompt, medium_type="image", use_audio_in_video=False)
        print(f"Response #{idx}: {response}\n")

        with open(file_path, "a") as f:
            f.write(f"Response #{idx}: {response}\n")
            if response != answer:
                f.write(f"WRONG! GT #{idx}: {answer}\n")
            else:
                correct += 1
    with open(file_path, "a") as f:
        f.write(f"Correctness: {correct}/{len(small_eval_dataset)} = {correct/len(small_eval_dataset):.2%}\n")
        print(f"Test Set Correctness: {correct}/{len(small_eval_dataset)} = {correct/len(small_eval_dataset):.2%}\n")


if __name__ == "__main__":
    main()
