#!/usr/bin/env python
"""
IPAdapter SDXL training/finetuning
"""
import argparse
import ast
import os
import sys
import time
from functools import partial

sys.path.append("../stable_diffusion_xl/")
sys.path.append("../stable_diffusion_v2/")

from gm.data.loader import create_loader
from gm.helpers import (
    create_model,
    get_grad_reducer,
    get_learning_rate,
    get_loss_scaler,
    get_optimizer,
    save_checkpoint,
    set_default,
)
from gm.models.trainer_factory import TrainerMultiGraphTwoStage, TrainOneStepCell
from gm.util.util import auto_mixed_precision
from ldm.util import count_params
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import Tensor, nn


def get_parser_train():
    parser = argparse.ArgumentParser(description="train with sd-xl")
    parser.add_argument("--version", type=str, default="SDXL-base-1.0", choices=["SDXL-base-1.0"])
    parser.add_argument("--config", type=str, default="configs/training/sd_xl_base_finetune_910b.yaml")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--clip_grad", default=False, type=ast.literal_eval, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--weight", type=str, default="checkpoints/sd_xl_base_1.0_ms_ip_adapter.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sd_xl_base_ratios", type=str, default="1.0")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./runs")
    parser.add_argument("--save_path_with_time", type=ast.literal_eval, default=True)
    parser.add_argument("--log_interval", type=int, default=1, help="log interval")
    parser.add_argument("--save_ckpt_interval", type=int, default=1000, help="save ckpt interval")
    parser.add_argument("--data_sink", type=ast.literal_eval, default=False)
    parser.add_argument("--sink_size", type=int, default=1000)
    parser.add_argument("--save_ip_only", type=ast.literal_eval, default=True)

    # args for env
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)"
    )
    parser.add_argument("--ms_amp_level", type=str, default="O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    parser.add_argument("--param_fp16", type=ast.literal_eval, default=False)
    parser.add_argument("--overflow_still_update", type=ast.literal_eval, default=True)
    parser.add_argument("--max_device_memory", type=str, default=None)
    parser.add_argument("--is_parallel", type=ast.literal_eval, default=False)

    # args for ModelArts
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument(
        "--ckpt_url", type=str, default="", help="ModelArts: obs path to pretrain model checkpoint file"
    )
    parser.add_argument("--train_url", type=str, default="", help="ModelArts: obs path to output folder")
    parser.add_argument(
        "--multi_data_url", type=str, default="", help="ModelArts: list of obs paths to multi-dataset folders"
    )
    parser.add_argument(
        "--pretrain_url", type=str, default="", help="ModelArts: list of obs paths to multi-pretrain model files"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/cache/pretrain_ckpt/",
        help="ModelArts: local device path to checkpoint folder",
    )
    return parser


def train(args):
    # 1. Init Env
    # manually set some args which are not used here but required by sd-xl
    args.dataset_load_tokenizer = True
    args.infer_interval = -1
    args = set_default(args)
    ms.set_context(ascend_config=dict(precision_mode="must_keep_origin_dtype"))

    # 2. Create LDM Engine
    config = OmegaConf.load(args.config)
    config_base = OmegaConf.load(config.pop("base", ""))
    config_base.merge_with(config)
    config = config_base

    model, _ = create_model(
        config,
        checkpoints=args.weight,
        freeze=False,
        load_filter=False,
        param_fp16=args.param_fp16,
        amp_level=args.ms_amp_level,
    )
    if isinstance(model.model, nn.Cell):
        model.model.set_train(True)
        model.conditioner.embedders[2].image_proj.set_train(True)

    # update the ip adapter paramters only
    for name, p in model.parameters_and_names():
        ip_names = ["to_k_ip", "to_v_ip", "image_proj"]
        if any([x in name for x in ip_names]):
            p.requires_grad = True
        else:
            p.requires_grad = False

    # 3. Create dataloader
    assert "data" in config
    dataloader = create_loader(
        data_path=args.data_path,
        rank=args.rank,
        rank_size=args.rank_size,
        tokenizer=model.conditioner.tokenize,
        token_nums=len(model.conditioner.embedders),
        **config.data,
    )

    # 4. Create train step func
    assert "optim" in config
    lr = get_learning_rate(config.optim, config.data.total_step)
    scaler = get_loss_scaler(ms_loss_scaler="static", scale_value=1024)
    if isinstance(model.model, nn.Cell):
        optimizer = get_optimizer(
            config.optim, lr, params=model.model.trainable_params() + model.conditioner.trainable_params()
        )
        reducer = get_grad_reducer(is_parallel=args.is_parallel, parameters=optimizer.parameters)
    else:
        optimizer, reducer = None, None

    if args.ms_mode == 1:
        # Pynative Mode
        assert isinstance(model.model, nn.Cell)
        train_step_fn = partial(
            model.train_step_pynative,
            grad_func=model.get_grad_func(
                optimizer, reducer, scaler, jit=True, overflow_still_update=args.overflow_still_update
            ),
        )
        model = auto_mixed_precision(model, args.ms_amp_level)
        jit_config = None
    elif args.ms_mode == 0:
        # Graph Mode
        if isinstance(model.model, nn.Cell):
            train_step_fn = TrainOneStepCell(
                model,
                optimizer,
                reducer,
                scaler,
                overflow_still_update=args.overflow_still_update,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                clip_grad=args.clip_grad,
                clip_norm=args.max_grad_norm,
            )
            train_step_fn = auto_mixed_precision(train_step_fn, amp_level=args.ms_amp_level)
            if model.disable_first_stage_amp:
                train_step_fn.first_stage_model.to_float(ms.float32)
            jit_config = ms.JitConfig()
        else:
            assert args.version == "SDXL-base-1.0", "Only supports sdxl-base."
            assert args.task == "txt2img", "Only supports text2img task."
            assert (model.stage1 is not None) and (model.stage2 is not None)
            optimizer1 = get_optimizer(
                config.optim, lr, params=model.conditioner.trainable_params() + model.stage1.trainable_params()
            )
            optimizer2 = get_optimizer(config.optim, lr, params=model.stage2.trainable_params())
            reducer1 = get_grad_reducer(is_parallel=args.is_parallel, parameters=optimizer1.parameters)
            reducer2 = get_grad_reducer(is_parallel=args.is_parallel, parameters=optimizer2.parameters)
            train_step_fn = TrainerMultiGraphTwoStage(
                model,
                (optimizer1, optimizer2),
                (reducer1, reducer2),
                scaler,
                overflow_still_update=args.overflow_still_update,
                amp_level=args.ms_amp_level,
            )

            optimizer = optimizer1
            jit_config = None
    else:
        raise ValueError("args.ms_mode value must in [0, 1]")

    num_params, num_trainable_params = count_params(model)
    print(f"Total number of parameters: {num_params:,}")
    print(f"Total number of trainable parameters: {num_trainable_params:,}")

    # 5. Start Training
    train_fn = train_txt2img if not args.data_sink else train_txt2img_datasink
    train_fn(args, train_step_fn, dataloader=dataloader, optimizer=optimizer, model=model, jit_config=jit_config)


def train_txt2img(args, train_step_fn, dataloader, optimizer=None, model=None, **kwargs):
    total_step = dataloader.get_dataset_size()
    loader = dataloader.create_tuple_iterator(output_numpy=True, num_epochs=1)
    s_time = time.time()
    for i, data in enumerate(loader):
        image, tokens = data[0], data[1:]
        image, tokens = Tensor(image), [Tensor(t) for t in tokens]

        # Train a step
        if i == 0:
            print(
                "The first step will be compiled for the graph, which may take a long time; "
                "You can come back later :)",
                flush=True,
            )
        loss, overflow = train_step_fn(image, *tokens)

        # Print meg
        if (i + 1) % args.log_interval == 0 and args.rank % 8 == 0:
            if optimizer.dynamic_lr:
                cur_lr = optimizer.learning_rate(Tensor(i, ms.int32)).asnumpy().item()
            else:
                cur_lr = optimizer.learning_rate.asnumpy().item()
            print(
                f"Step {i + 1}/{total_step}, size: {image.shape[2:]}, lr: {cur_lr}, loss: {loss.asnumpy():.6f}"
                f", time cost: {(time.time()-s_time) * 1000 / args.log_interval:.2f} ms",
                flush=True,
            )
            s_time = time.time()

        # Save checkpoint
        if (i + 1) % args.save_ckpt_interval == 0 and args.rank % 8 == 0:
            save_ckpt_dir = os.path.join(args.save_path, "weights", args.version + f"_{(i + 1)}.ckpt")
            if isinstance(model.model, nn.Cell):
                model.model.set_train(False)
                model.conditioner.embedders[2].image_proj.set_train(False)
                if args.save_ip_only:
                    save_ckpt_dir = save_ckpt_dir.replace(".ckpt", "_ip_only.ckpt")
                    save_ip_checkpoint(model, save_ckpt_dir)
                else:
                    save_checkpoint(model, save_ckpt_dir)
                model.model.set_train(True)
                model.conditioner.embedders[2].image_proj.set_train(True)
            else:
                model.save_checkpoint(save_ckpt_dir)


def train_txt2img_datasink(args, train_step_fn, dataloader, optimizer=None, model=None, jit_config=None, **kwargs):
    total_step = dataloader.get_dataset_size()
    epochs = total_step // args.sink_size

    train_fn_sink = ms.data_sink(fn=train_step_fn, dataset=dataloader, sink_size=args.sink_size, jit_config=jit_config)

    for epoch in range(epochs):
        cur_step = args.sink_size * (epoch + 1)

        if epoch == 0:
            print(
                "The first epoch will be compiled for the graph, which may take a long time; "
                "You can come back later :)",
                flush=True,
            )

        s_time = time.time()
        loss, _ = train_fn_sink()
        e_time = time.time()

        # Print meg
        if cur_step % args.log_interval == 0 and args.rank % 8 == 0:
            if optimizer.dynamic_lr:
                cur_lr = optimizer.learning_rate(Tensor((cur_step - 1), ms.int32)).asnumpy().item()
            else:
                cur_lr = optimizer.learning_rate.asnumpy().item()
            print(
                f"Step {cur_step}/{total_step}, lr: {cur_lr}, loss: {loss.asnumpy():.6f}"
                f", per step time: {(e_time - s_time) * 1000 / args.sink_size:.2f} ms",
                flush=True,
            )

        # Save checkpoint
        if cur_step % args.save_ckpt_interval == 0 and args.rank % 8 == 0:
            save_ckpt_dir = os.path.join(args.save_path, "weights", args.version + f"_{cur_step}.ckpt")
            if isinstance(model.model, nn.Cell):
                model.model.set_train(False)
                model.conditioner.embedders[2].image_proj.set_train(False)
                if args.save_ip_only:
                    save_ckpt_dir = save_ckpt_dir.replace(".ckpt", "_ip_only.ckpt")
                    save_ip_checkpoint(model, save_ckpt_dir)
                else:
                    save_checkpoint(model, save_ckpt_dir)
                model.model.set_train(True)
                model.conditioner.embedders[2].image_proj.set_train(True)
            else:
                model.save_checkpoint(save_ckpt_dir)


def save_ip_checkpoint(model: nn.Cell, path: str):
    ckpt_ip = []
    ip_names = ["to_k_ip", "to_v_ip", "image_proj"]
    for name, param in model.parameters_and_names():
        if any([x in name for x in ip_names]):
            # remove strange prefix
            if "._backbone" in name:
                name = name.replace("._backbone", "")
            ckpt_ip.append({"name": name, "data": param})

    ms.save_checkpoint(ckpt_ip, path)
    print(f"save checkpoint to {path}")


if __name__ == "__main__":
    parser = get_parser_train()
    args, _ = parser.parse_known_args()
    train(args)
