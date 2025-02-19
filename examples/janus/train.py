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
    
    # 1.3 save the model config
    config.save_pretrained(args.output_path)

    # 2 model with loss
    # integarte in vl_gpt, modeling_vlm

    # 3. create dataset

    # 4. build training utils: lr, optim, callbacks, trainer
    # build learning rate scheduler

    set_all_reduce_fusion(
        latent_diffusion_with_loss.trainable_params(),
        split_num=7,
        distributed=args.use_parallel,
        parallel_mode=args.parallel_mode,
    )

    # build optimizer
    optimizer = create_optimizer(
        vl_gpt.trainable_params(),
        name="adamw_re",  # FIXME
        group_strategy=None,  # FIXME
        weight_decay=None, # FIXME
        lr=args.learning_rate,
    )
    
    loss_scaler = nn.FixedLossScaleUpdateCell(1024)  # FIXME

    # resume ckpt
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    start_epoch = 0
    cur_iter = 0
    if args.resume:
        resume_ckpt = get_resume_ckpt(args.resume, args.output_path)
        if resume_ckpt is not None:
            start_epoch, cur_iter, loss_scale = get_resume_states(resume_ckpt)
            loss_scaler.loss_scale_value = loss_scale
            logger.info(f"Resumed loss_scaler, prev epoch: {start_epoch}, global step {cur_iter}")

    # train one step (standalone and distributed)
    net_with_grads = prepare_train_network(
        vl_gpt,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=True,  # FIXME
        clip_grad=False,  # FIXME
        clip_norm=1.0,    # FIXME
        # ema=ema,
        # zero_stage=args.zero_stage,
    )

    # resume train net states
    # if args.resume and resume_ckpt is not None:
    #    resume_train_net(net_with_grads, resume_ckpt)

    # 5. log and save config
    if rank_id == 0:
        logger.info("Start training...")

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    # 6. train
    global_step = cur_iter  # index start from 1 (after first-step network update)

    if rank_id == 0:
        ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        perf_columns = ["step", "loss", "train_time(s)", "shape"]
        output_dir = ckpt_dir.replace("/ckpt", "")
        if start_epoch == 0:
            record = PerfRecorder(output_dir, metric_names=perf_columns)
        else:
            record = PerfRecorder(output_dir, resume=True)

    # ds_iter = dataloader.create_tuple_iterator(num_epochs=num_epochs - start_epoch)
    # ds_iter = dataloader.create_tuple_iterator(num_epochs=-1) # infinite

    from tests.test_toy_data import  gen_t2i_train_sample
    input_ids, labels, attention_masks, image, image_seq_masks = gen_t2i_train_sample(max_length=1024)
    sample = (ms.Tensor(input_ids), ms.Tensor(labels), ms.Tensor(attention_masks), ms.Tensor(image), ms.Tensor(image_seq_masks))

    end_train = False
    num_epochs = args.train_steps
    for epoch in range(start_epoch + 1, num_epochs + 1):
        if (args.train_steps > 0) and (global_step >= args.train_steps):
            logger.warning("resumed steps >= train_steps, will end training")
            break

        start_time_s = time.time()
        # for step, data in enumerate(ds_iter, 1):
        for step in range(3):
            loss, overflow, scaling_sens = net_with_grads(*sample)
            global_step += 1
            step_time = time.time() - start_time_s

            loss_val = float(loss.asnumpy())
            logger.info(
                f"Epoch {epoch}, Step {step}, loss {loss_val:.5f}, Global step {global_step},"
                + f" Shape: {tuple(data[0].shape)}, Step time {step_time*1000:.2f}ms"
            )
            if overflow:
                logger.warning("overflow detected")

            if rank_id == 0:
                step_pref_value = [global_step, loss_val, step_time, tuple(data[0].shape)]
                record.add(*step_pref_value)
            # save and eval in step
            if save_by_step and rank_id == 0:
                if (global_step % args.ckpt_save_steps == 0) or (global_step == args.train_steps):
                    ckpt_name = f"model-s{global_step}.ckpt"
                    # save model ckpt and ema ckpt
                    save_ema_ckpts(latent_diffusion_with_loss.network, ema, ckpt_manager, ckpt_name)
                    # save train state for resume
                    save_train_net(net_with_grads, ckpt_dir, epoch - 1, global_step)
            if (args.train_steps > 0) and (global_step >= args.train_steps):
                end_train = True
                break

            start_time_s = time.time()

        # dataloader.reset()
        flush_from_cache(net_with_grads)

        if end_train:
            break

    logger.info("Finished training. Ending process...")
    reset_op_id()
    # time.sleep(60)
    logger.info("End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_mode", type=int, default=1, help="mindspore mode, 0: graph, 1: pynative")
    # TODO: support model_name "deepseek-ai/Janus-Pro-1B" for simplicity
    # FIXME: change to deepseek-ai/Janus-Pro-1B after debug
    parser.add_argument("--model_path", type=str, default="ckpts/lite", help="path to Janus model")
    parser.add_argument("--training_stage", type=int, default=3, choices=[1, 2, 3], help="model training stage, can be 1, 2, or 3")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to model checkpoint in .ckpt format, if None, will use the pretrained weight in mode_path")
    # FIXME: change to False after debug
    parser.add_argument("--load_weight", type=str2bool, default=False, help="if True, will not load pretrained weight in model_path")
    parser.add_argument("--dtype", type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'], help="model dtype")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")

    # training hyperparms
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--train_steps", default=5000, type=int, help="training step")
    parser.add_argument("--ckpt_save_steps", default=100, type=int, help="save ckpt every this step")

    args = parser.parse_args()

    main(args)
