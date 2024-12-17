import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from types import MethodType
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, context, dataset, nn, ops
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.train.amp import AMP_BLACK_LIST, _auto_black_list

# init()
# rank, rank_size, parallel_mode = get_rank(), get_group_size(), context.ParallelMode.DATA_PARALLEL
# context.set_auto_parallel_context(
#     device_num=rank_size, parallel_mode=parallel_mode, gradients_mean=True
# )

rank, rank_size = 0, 1

ms.set_context(mode=ms.context.PYNATIVE_MODE, pynative_synchronize=True, mempool_block_size="59GB", max_device_memory="59GB")

import transformers
from transformers import HfArgumentParser

from mindspore.dataset import transforms, vision

# from accelerate.utils import DistributedType

mindone_lib_path = os.path.abspath(os.path.abspath("../../../"))
sys.path.insert(0, mindone_lib_path)

from dataset import SupervisedDataset
from mindone.transformers.trainer import Trainer
from transformers import AutoTokenizer
from mindone.transformers.training_args import TrainingArguments

from mindone.transformers.models.minicpm_v2_6 import MiniCPMV_v2_6
from mindone.transformers.mindspore_adapter import MindSporeArguments

# from transformers.integrations import deepspeed


# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ms.set_context(mode=ms.context.PYNATIVE_MODE, pynative_synchronize=True)
# ms.set_context(mode=ms.context.PYNATIVE_MODE)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-V-2")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )

# @dataclass
# class TrainingArguments(TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_mindspore")
#     model_max_length: int = field(
#         default=2048,
#         metadata={
#             "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
#         },
#     )
#     tune_vision: Optional[bool] = field(default=True)
#     tune_llm: Optional[bool] = field(default=True)
#     llm_type: str = field(default="minicpm")
#     use_lora: Optional[bool] = field(default=False)
#     max_slice_nums: Optional[int] = field(default=9)
#     distributed: Optional[bool] = field(default=False)
#     amp_level: Optional[str] = field(default="O0")


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None

@dataclass
class MyArguments(MindSporeArguments, TrainingArguments):
    enable_flash_attention: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    is_distribute: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_mindspore")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    llm_type: str = field(default="minicpm")
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)
    distributed: Optional[bool] = field(default=False)
    amp_level: Optional[str] = field(default="O0")

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer.save_model(output_dir,)

# class ModifiedMapFunction(BaseMapFuction):
#     def __call__(self, input_ids, position_ids, labels, attention_mask):
#         return trim_and_pad(input_ids), trim_and_pad(position_ids), trim_and_pad(labels), trim_and_pad(attention_mask)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    transform,
    data_collator=None,
    llm_type="minicpm",
    slice_config=None,
    patch_size=14,
    query_nums=64,
    batch_vision=False,
    max_length=2048,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    dataset_cls = SupervisedDataset

    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(
        train_json,
        transform,
        tokenizer,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
        max_length=max_length,
    )

    # train_ds = dataset.GeneratorDataset(
    #     train_dataset,
    #     column_names=train_dataset.dataset_column_names,
    #     num_parallel_workers=2,
    #     shuffle=True,
    #     python_multiprocessing=False,
    #     num_shards=rank_size,
    #     shard_id=rank
    # )

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(
            eval_json,
            transform,
            tokenizer,
            slice_config=slice_config,
            llm_type=llm_type,
            patch_size=patch_size,
            query_nums=query_nums,
            batch_vision=batch_vision,
            max_length=max_length,
        )

        # eval_ds = dataset.GeneratorDataset(
        #     eval_dataset,
        #     column_names=eval_dataset.dataset_column_names,
        #     num_parallel_workers=8,
        #     shuffle=False,
        #     python_multiprocessing=False,
        # )
    else:
        eval_dataset = None

    # def trim_and_pad(seq):
    #     # return pad_sequence([s[:max_length] for s in seq], batch_first=True, padding_value=padding_value)
    #     max_length = 2048
    #     return np.stack([s[:max_length] for s in seq])
    #
    # class ModifiedMapFunction(BaseMapFuction):
    #     def __call__(self, input_ids, position_ids, labels, attention_mask):
    #         return trim_and_pad(input_ids), trim_and_pad(position_ids), trim_and_pad(labels), trim_and_pad(attention_mask)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


# def build_transform():
#     IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
#     IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
#     return transforms.Compose(
#             [
#                 vision.ToTensor(),
#                 vision.Normalize(
#                     mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, is_hwc=False
#                 ),
#             ]
#         )

def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
            [
                vision.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, is_hwc=False
                ),
            ]
        )

def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    # for param in model.parameters():
    #     num_params = param.numel()
    #     # if using DS Zero 3 and the weights are initialized empty
    #     if num_params == 0 and hasattr(param, "ds_numel"):
    #         num_params = param.ds_numel
    #
    #     all_param += num_params
    #     if param.requires_grad:
    #         trainable_params += num_params
    for param in model.trainable_params():
        num_params = np.prod(param.shape)
        trainable_params += num_params

    return {'Trainable params': trainable_params}


local_rank = 0


def train():
    global local_rank
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, MyArguments, LoraArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # if getattr(training_args, "deepspeed", None) :
    #     training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        ms.float16
        if training_args.fp16
        else (ms.bfloat16 if training_args.bf16 else ms.float32)
    )

    # if training_args.distributed:
    #     init()
    #     data_args.rank, data_args.rank_size, parallel_mode = get_rank(), get_group_size(), context.ParallelMode.DATA_PARALLEL
    #     context.set_auto_parallel_context(
    #         device_num=data_args.rank_size, parallel_mode=parallel_mode, gradients_mean=True
    #     )
    # else:
    #     data_args.rank, data_args.rank_size, parallel_mode = 0, 1, None

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0:
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    model = MiniCPMV_v2_6.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        mindspore_dtype=compute_dtype,
    )

    if training_args.amp_level == "O2":
        _auto_black_list(
            model,
            AMP_BLACK_LIST + [nn.GroupNorm, nn.SiLU],
            ms.float16,
        )
    elif training_args.amp_level == "O3":
        model.to_float(ms.float16)

    # if training_args.distributed:
    #     # set grad reducer
    #     mean = ms.context.get_auto_parallel_context("gradients_mean")
    #     degree = ms.context.get_auto_parallel_context("device_num")
    #     grad_reducer = nn.DistributedGradReducer(model.trainable_params(), mean, degree)
    # else:
    #     grad_reducer = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    if not training_args.tune_vision:
        # model.vpm.set_train(False)
        for param in model.vpm.trainable_params():
            param.requires_grad = False
    if not training_args.tune_llm:
        # model.llm.set_train(False)
        for param in model.llm.trainable_params():
            param.requires_grad = False

    if training_args.use_lora:
        if training_args.use_lora and training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")

        rank0_print("Currently using LoRA for fine-tuning the MiniCPM-V model.")
        for name, param in model.llm.named_parameters():
            param.requires_grad = False
        modules_to_save = ['embed_tokens','resampler']
        if training_args.tune_vision:
            modules_to_save.append('vpm')
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
        if not hasattr(model, 'get_input_embeddings'):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    rank0_print(get_parameter_number(model))

    llm_type = training_args.llm_type

    rank0_print(f'llm_type={llm_type}')


    # Load data
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.to_dict()

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False

    transform_func = build_transform()
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        transform=transform_func,
        data_collator=None,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=model.config.patch_size,
        query_nums=model.config.query_num,
        batch_vision=batch_vision,
        max_length=training_args.model_max_length,
    )

    training_args.gradient_checkpointing_kwargs={"use_reentrant":False}
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train()
    # trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
