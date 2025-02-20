"""
Reproduce JanusPro training
"""
import argparse
import datetime
import logging
import math
import os
import sys
import time
from typing import Optional, Tuple

import yaml

import mindspore as ms
from mindspore import nn
from mindspore._c_expression import reset_op_id
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.models.modeling_vlm import MultiModalityConfig
from janus.utils.io import set_model_param_dtype

from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallbackEpoch, StopAtStepCallback
from mindone.trainers.checkpoint import CheckpointManager
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.recorder import PerfRecorder
from mindone.trainers.zero import prepare_train_network
from mindone.trainers.train_step import TrainOneStepWrapper
# from mindone.transformers.mindspore_adapter import HF2MSDataset, TrainOneStepWrapper, auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindone.utils.seed import set_random_seed
from mindone.utils.config import str2bool


logger = logging.getLogger(__name__)


def init_env(mode, seed, distribute=False):
    set_random_seed(seed)
    # ms.set_context(max_device_memory=max_device_memory)
    ms.set_context(mode=mode)
    ms.set_context(jit_config={"jit_level": "O0"})

    if distribute:
        ms.set_context(mode=mode)
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
        ms.reset_auto_parallel_context()

        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        device_num = 1
        rank_id = 0
        ms.set_context(mode=mode)

    return rank_id, device_num


def main(args):
    # 0. env init
    # time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # args.output_path = os.path.join(args.output_path, time_str)

    rank_id, device_num = init_env(
        args.ms_mode,
        args.seed,
        distribute=args.use_parallel,
        )

    set_logger(name="", output_dir=args.output_path, rank=rank_id)

    # 1. janus model init
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    config =  MultiModalityConfig.from_pretrained(args.model_path)
    config.torch_dtype = args.dtype
    config.language_config.torch_dtype = args.dtype
    config.language_config._attn_implementation = 'flash_attention_2'  # use FA by default
    if args.load_weight:
        vl_gpt = MultiModalityCausalLM.from_pretrained(args.model_path, config=config)
    else:
        vl_gpt = MultiModalityCausalLM(config=config)
    if args.ckpt_path is not None:
        parameter_dict = ms.load_checkpoint(args.ckpt_path)
        param_not_load, ckpt_not_load = ms.load_param_into_net(vl_gpt, parameter_dict, strict_load=True)
        logger.info("net param not load: ".format(param_not_load))
        logger.info("ckpt param not load: ".format(ckpt_not_load))

    # 1.1 mixed precision
    dtype_map = {"float16": ms.float16, "bfloat16": ms.bfloat16} 
    dtype = dtype_map[args.dtype]
    if args.dtype != "float32":
        vl_gpt = set_model_param_dtype(vl_gpt, dtype)
    
    # 1.2 set trainable parameters (refer to Janus paper)
    # TODO: use config.yaml to set traning strategy
    num_frozen_params = 0 
    num_train_params = 0 
    all_modules = set([vl_gpt.vision_model, vl_gpt.gen_vision_model, vl_gpt.language_model, vl_gpt.aligner, vl_gpt.gen_aligner, vl_gpt.gen_head, vl_gpt.gen_embed])
    if args.training_stage == 1:
        # Stage I: Training adaptors and image head
        # freeze sigLIP, VQ16, llm; train adaptors and image head
        frozen_modules = set([vl_gpt.vision_model, vl_gpt.gen_vision_model, vl_gpt.language_model])
    elif args.training_stage == 2:
        # Stage II: unfied pretraining
        # further unfreeze llm 
        frozen_modules = set([vl_gpt.vision_model, vl_gpt.gen_vision_model])
    elif args.training_stage == 3:
        # Stage III: SFT
        # only freeze gen. vision autoencoder(VQ); train all others: gen adaptor, und. enc (sliLIP) + und. adaptor, LLM, text head, image head   
        # TODO: whether gen_embed (nn.Embed) should be trainable in stage 3
        frozen_modules = set([vl_gpt.gen_vision_model])
    else:
        raise NotImplementedError
    # VQ encoder doesn't need grad
    vl_gpt.gen_vision_model.set_grad(requires_grad=False)
    trainable_modules = all_modules - frozen_modules
    
    for module in frozen_modules:
        module.set_train(False)
        for param in module.get_parameters():
            param.requires_grad = False 
            num_frozen_params += 1

    for module in trainable_modules:
        module.set_train(True)
        for param in module.get_parameters():
            param.requires_grad = True 
            num_train_params += 1

    tot_params = len(list(vl_gpt.get_parameters()))
    print(f'tot params: {tot_params}, trainable params: {num_train_params}, frozen params: {num_frozen_params}')
    assert num_frozen_params + num_train_params == tot_params, 'All params should be set to trainable or frozen.'
    # 1.2  prepare dataset
    # FIXME: add dataset and loader. this is a toy data sample for i2v debug
    from tests.test_toy_data import  gen_t2i_train_sample
    input_ids, labels, attention_masks, image_seq_masks, image = gen_t2i_train_sample(max_length=args.max_length)
    
    # 1.3 save the model config
    config.save_pretrained(args.output_path)
    
    '''
    lr = create_scheduler(
        steps_per_epoch=-1,
        name="cosine_decay",
        lr=1e-5,
        end_lr=1e-6,
        warmup_steps=30,
        # decay_steps=args.decay_steps,
        total_steps=args.train_steps,
    )
    loss_scaler = nn.FixedLossScaleUpdateCell(1024)  # FIXME
    '''
    
    # 3
    # hyper params refer to emu3 sft.
    # FIXME:  use cosine_with_min_lr w/ lr=1e-5 min=1e-6, but mint adamw don't support lr list.
    optimizer = ms.mint.optim.AdamW(vl_gpt.trainable_params(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-6,
        )

    train_step = TrainOneStepWrapper(
        vl_gpt,
        optimizer=optimizer,
        scale_sense=ms.Tensor(1.0),
        clip_grad=True,  # FIXME
        clip_norm=1.0,    # FIXME
        # ema=ema,
        # zero_stage=args.zero_stage,
    )
     
    # FIXME: for sequence parallel, save ckpt for other ranks
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    # TODO: suppor training resume 
    start_epoch = 0
    if rank_id == 0:
        ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=2)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        perf_columns = ["step", "loss", "train_time(s)"]
        output_dir = ckpt_dir.replace("/ckpt", "")
        if start_epoch == 0:
            record = PerfRecorder(output_dir, metric_names=perf_columns)
        else:
            record = PerfRecorder(output_dir, resume=True)

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

        logger.info("Start training...")

    # training loop
    start_time_s = time.time()
    for step in range(args.train_steps): 
        data = (ms.Tensor(input_ids, dtype=ms.int32),
            ms.Tensor(labels, dtype=ms.int32),
            ms.Tensor(attention_masks, dtype=ms.bool_),
            ms.Tensor(image_seq_masks, dtype=ms.bool_),
            ms.Tensor(image, dtype=dtype),
            )

        # loss = train_step(data)
        loss, overflow, scaling_sens = train_step(*data) 

        step_time = time.time() - start_time_s
        loss_val = float(loss.asnumpy())
        logger.info(
            f"Step {step}, loss {loss_val:.5f}, step time {step_time*1000:.2f}ms"
        )
        if rank_id == 0:
            step_pref_value = [step, loss_val, step_time]
            record.add(*step_pref_value)

        if (step > 0) and (step  % 500 == 0): 
            # ms.save_checkpoint(vl_gpt, os.path.join(args.output_path, "janus_ft.ckpt"))
            ckpt_name = f"model-s{step}.ckpt"
            ckpt_manager.save(vl_gpt, None, ckpt_name=ckpt_name, append_dict=None)
        start_time_s = time.time()
    
    logger.info("Finished training. Ending process...")
    reset_op_id()
    logger.info("End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_mode", type=int, default=1, help="mindspore mode, 0: graph, 1: pynative")
    # TODO: support model_name "deepseek-ai/Janus-Pro-1B" for simplicity
    parser.add_argument("--model_path", type=str, default="ckpts/Janus-Pro-1B", help="path to Janus model")
    parser.add_argument("--training_stage", type=int, default=3, choices=[1, 2, 3], help="model training stage, can be 1, 2, or 3")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to model checkpoint in .ckpt format, if None, will use the pretrained weight in mode_path")
    parser.add_argument("--load_weight", type=str2bool, default=True, help="if True, will not load pretrained weight in model_path")
    parser.add_argument("--dtype", type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'], help="model dtype")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--output_path", default="outputs/janus-sft", type=str, help="output directory to save training results")

    # training hyperparms
    parser.add_argument("--learning_rate", default=5e-6, type=float, help="learning rate")
    parser.add_argument("--train_steps", default=5000, type=int, help="training step")
    parser.add_argument("--ckpt_save_steps", default=100, type=int, help="save ckpt every this step")
    parser.add_argument("--max_length", default=1024, type=int, help="sequence max length, input sequence will be padded to this max length")

    args = parser.parse_args()

    main(args)
