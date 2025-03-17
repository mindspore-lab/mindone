import logging
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import transformers as tf
from emu3.acceleration import create_parallel_group
from emu3.mllm import Emu3Config, Emu3ForCausalLM, Emu3Tokenizer
from emu3.mllm.modeling_emu3 import Emu3RMSNorm
from emu3.train.datasets import Emu3FeatureDataset

import mindspore as ms
import mindspore.dataset as ds
from mindspore import Model, nn  # amp
from mindspore.nn import CrossEntropyLoss
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

from mindone.data import create_dataloader
from mindone.trainers import create_optimizer
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, StopAtStepCallback
from mindone.trainers.zero import prepare_train_network
from mindone.transformers.mindspore_adapter import MindSporeArguments
from mindone.transformers.optimization import get_scheduler
from mindone.transformers.training_args import TrainingArguments as tf_TrainingArguments
from mindone.utils import count_params, init_train_env, set_logger
from mindone.utils.amp import auto_mixed_precision

# from mindone.trainers.checkpoint import resume_train_network


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Stage1")  # Emu3-Gen/Chat/Stage1


@dataclass
class DataArguments:
    train_data_path: Optional[str] = field(default=None)
    eval_data_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)


@dataclass
class TrainingArguments(MindSporeArguments, tf_TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    image_area: Optional[int] = field(default=None)  # image max resolution, e.g. 518440=720*720, 262144=512*512
    max_position_embeddings: Optional[int] = field(default=None)
    trainable_hidden_layers: Optional[int] = field(default=32)
    output_dir: str = field(default="./outputs")  # output directory for checkpoints
    enable_flash_attention: bool = field(default=True)  # enable flash_attention_2
    gradient_checkpointing: bool = field(default=True)  # activate gradient checkpointing
    is_distribute: bool = field(default=False)  # use data parallel
    debug: bool = field(default=False)  # enable pynative synchronize for debugging
    seed: int = field(default=42)
    sequence_parallel_shards: int = field(default=1)  # number of sequential parallelism shards
    ms_zero_stage: int = field(default=0)
    resume: str = field(default=None)
    model_name: str = field(default="emu3")
    clip_grad: bool = field(default=True)
    loss_scaler_type: str = field(default=None)  # dynamic or static
    init_loss_scale: float = field(default=1024)
    loss_scale_factor: float = field(default=2)
    scale_window: float = field(default=1000)
    save_strategy: str = field(default="steps")
    # post_init_weight: bool = field(default=False)


def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None
        else setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)


# def resume_train_net(
#     train_net: TrainOneStepWrapper, resume_ckpt = None
# ) -> Tuple[Union[int, None], Union[int, None]]:
#     if resume_ckpt is None:
#         return None, None

#     state_dict = ms.load_checkpoint(resume_ckpt)
#     if "epoch_num" not in state_dict or "cur_step" not in state_dict or "loss_scale" not in state_dict:
#         raise ValueError("Resume training checkpoint is invalid. Please check the checkpoint file.")

#     start_epoch = state_dict.pop("epoch_num").item()
#     global_step = state_dict.pop("cur_step").item()
#     logger.info(f"Resuming training of network from {resume_ckpt} at global step {global_step}")

#     # `EvalSaveCallback` renames `scale_sense` to `loss_scale` when saving the resume checkpoint
#     train_net.scale_sense = ms.Parameter(state_dict.pop("loss_scale"), name="scale_sense")
#     param_not_load, ckpt_not_load = load_param_into_net_with_filter(train_net, state_dict, filter=state_dict.keys())
#     if param_not_load or ckpt_not_load:
#         logger.warning(
#             f"Exist ckpt params not loaded: {ckpt_not_load} (total: {len(ckpt_not_load)}),\n"
#             f"or net params not loaded: {param_not_load} (total: {len(param_not_load)})"
#         )

#     return start_epoch, global_step


def main():
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1.0. init training env
    device_id, rank_id, device_num = init_train_env(
        mode=training_args.mode,
        device_target=training_args.device_target,  # default: "Ascend"
        debug=training_args.debug,
        seed=training_args.seed,
        distributed=training_args.is_distribute,
        jit_level=training_args.jit_level,  # default: O0
        max_device_memory=training_args.max_device_memory,
    )
    set_logger("", output_dir=training_args.output_dir, rank=rank_id)

    # 1.1. init model parallelism
    shard_rank_id = rank_id
    if training_args.sequence_parallel_shards > 1:
        create_parallel_group(training_args.sequence_parallel_shards)
        device_num = device_num // training_args.sequence_parallel_shards
        shard_rank_id = rank_id // training_args.sequence_parallel_shards

    # set_seed(training_args.seed + shard_rank_id)  # set different seeds per NPU for sampling different timesteps
    ds.set_seed(training_args.seed)  # keep MS.dataset's seed consistent as datasets first shuffled and then distributed

    # 1. create dataset
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )

    train_dataset = Emu3FeatureDataset(data_args, tokenizer=tokenizer, split="train")
    dataset_len = len(train_dataset)
    num_update_steps_per_epoch = max(1, dataset_len // training_args.gradient_accumulation_steps)
    num_training_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)

    # TODO
    # eval_dataset = None
    # if (data_args.eval_data_path is not None) and training_args.do_eval:
    #     eval_dataset = Emu3FeatureDataset(data_args, tokenizer=tokenizer, split="test")

    batch_size = training_args.per_device_train_batch_size * device_num
    num_epoch = int(training_args.num_train_epochs)  # default=3
    if training_args.sequence_parallel_shards > 1:
        logger.info(
            f"Initializing the dataloader: assigning shard ID {shard_rank_id} out of {device_num} total shards."
        )
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        transforms=None,
        device_num=device_num,
        rank_id=shard_rank_id,
        shuffle=True,
        num_workers_dataset=training_args.dataloader_num_workers,
    )

    # 2. create train network and mix precision
    # load pre-trained model and tokenizer (usually Emu3-Stage1)
    model_config = Emu3Config.from_pretrained(model_args.model_name_or_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate

    # TODO: if resume
    model_dtype = ms.bfloat16 if training_args.bf16 else (ms.float16 if training_args.fp16 else None)
    # DEBUG
    # model_config.num_hidden_layers = 1
    # mdoel_config.attn_implementation="flash_attention_2"
    # model = Emu3ForCausalLM(model_config)
    # DEBUG
    model = Emu3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        mindspore_dtype=model_dtype,
        attn_implementation="flash_attention_2" if training_args.enable_flash_attention else None,
        use_safetensors=True,
    )
    # Optional. Some layers can be frozen to speed up trainining and reduce memory.
    if training_args.trainable_hidden_layers < model_config.num_hidden_layers:
        for layer_id in range(model_config.num_hidden_layers - training_args.trainable_hidden_layers):
            for param in model.model.layers[layer_id].get_parameters():
                param.requires_grad = False
    if model_dtype is not None:
        model = auto_mixed_precision(
            model, amp_level="O2", dtype=model_dtype, custom_fp32_cells=[CrossEntropyLoss, Emu3RMSNorm]
        )
    else:
        model_dtype = ms.float32

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    class TrainNetWithLoss(nn.Cell):
        def __init__(self, network):
            super(TrainNetWithLoss, self).__init__()
            self.network = network

        def construct(self, input_ids, attention_mask, labels):
            loss = self.network(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
            return loss

        def set_train(self, mode=True):
            self.network.set_train(mode)

    model_with_loss = TrainNetWithLoss(model)

    # 3. training setups: lr scheduler, optimizer, trainer, etc.
    # lr scheduler
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        base_lr=training_args.learning_rate,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
    )
    # optimizer
    optimizer_kwargs = {
        "name": "adamw",
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
        "lr": lr_scheduler,
    }
    optimizer = create_optimizer(model_with_loss.trainable_params(), **optimizer_kwargs)

    # trainer
    ema = None
    # ema = EMA(model_with_loss.network, **training_args.ema.init_args) if args.train.ema else None

    loss_scaler = None
    if training_args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=training_args.init_loss_scale,
            scale_factor=training_args.loss_scale_factor,
            scale_window=training_args.scale_window,
        )
    elif training_args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(training_args.init_loss_scale)
    else:
        raise ValueError

    net_with_grads = prepare_train_network(
        model_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        ema=ema,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        clip_grad=training_args.clip_grad,
        clip_norm=training_args.max_grad_norm,
        zero_stage=training_args.ms_zero_stage,
    )
    # net_with_grads = TrainOneStepWrapper(model_with_loss, optimizer)

    start_epoch, global_step = 0, 0

    # TODO
    # if training_args.resume is not None:
    #     logger.info(f"Loading train_resume.ckpt in {training_args.resume} to resume training")
    #     resume_ckpt = os.path.join(training_args.resume, "ckpt", "train_resume.ckpt")
    #     start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
    #         model_with_loss.network, optimizer, resume_ckpt
    #     )  # NOTE: if total training steps is different from original resume checkpoint, optimizer has different shape and encounter error.
    #     loss_scaler.loss_scale_value = loss_scale
    #     loss_scaler.cur_iter = cur_iter
    #     loss_scaler.last_overflow_iter = last_overflow_iter
    #     global_step = cur_iter

    # training model
    train_model = Model(net_with_grads)

    # callbacks
    callbacks = [OverflowMonitor()]
    if training_args.ms_zero_stage == 3 or rank_id == 0:
        ckpt_save_dir = (
            os.path.join(training_args.output_dir, f"rank_{rank_id}/ckpt")
            if training_args.ms_zero_stage == 3
            else os.path.join(training_args.output_dir, "ckpt")
        )
        callbacks.append(
            EvalSaveCallback(
                network=model_with_loss.network,
                model_name=training_args.model_name,
                rank_id=0 if training_args.ms_zero_stage == 3 else rank_id,  # ZeRO-3 shards across all ranks
                ckpt_save_dir=ckpt_save_dir,
                ema=ema,
                use_step_unit=True if training_args.save_strategy == "steps" else False,
                start_epoch=start_epoch,
                train_steps=num_training_steps,
                log_interval=training_args.logging_steps,
                ckpt_max_keep=training_args.save_total_limit,
                ckpt_save_interval=training_args.save_steps,
                step_mode=True if training_args.save_strategy == "steps" else False,  # epoch/steps, default: steps
            )
        )
    # if rank_id == 0:
    #     callbacks.append(
    #         PerfRecorderCallback(
    #             args.train.output_path, file_name="result_val.log", metric_names=["eval_loss", "eval_loss_smoothed"]
    #         )
    #     )
    callbacks.append(StopAtStepCallback(train_steps=num_training_steps, global_step=global_step))

    # print out key info and save config
    if rank_id == 0:
        num_params, num_params_trainable = count_params(model_with_loss)
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {training_args.mode}",
                f"Debug mode: {training_args.debug}",
                f"JIT level: {training_args.jit_level}",
                f"Distributed mode: {training_args.is_distribute}",
                f"Train data path: {data_args.train_data_path}",
                f"Val data path: {data_args.eval_data_path}",
                f"Number of samples: {dataset_len}",
                f"Model name: {training_args.model_name}",
                f"Model dtype: {model_dtype}",
                f"Num params: {num_params:,}",
                f"Num trainable params: {num_params_trainable:,}",
                f"Learning rate: {training_args.learning_rate:.0e}",
                f"Batch size: {batch_size}",
                f"Max image area: {training_args.image_area}",
                f"Max token length: {training_args.max_position_embeddings}",
                f"Grad accumulation steps: {training_args.gradient_accumulation_steps}",
                f"Number of training steps: {num_training_steps}",
                f"Loss scaler: {training_args.loss_scaler_type}",
                f"Init loss scale: {training_args.init_loss_scale}",
                f"Grad clipping: {training_args.clip_grad}",
                f"Max grad norm: {training_args.max_grad_norm}",
                f"EMA: {ema is not None}",
                f"Enable flash attention: {training_args.enable_flash_attention}",
            ]
        )
        key_info += "\n" + "=" * 50
        print(key_info)
        model_config.save_pretrained(save_directory=training_args.output_dir)

    # 4 .train
    logger.info("Start training...")
    # train() uses epochs, so the training will be terminated by the StopAtStepCallback
    train_model.train(num_epoch, train_dataloader, callbacks=callbacks, initial_epoch=start_epoch)


if __name__ == "__main__":
    main()
