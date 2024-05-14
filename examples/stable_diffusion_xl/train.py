import argparse
import ast
import os
import random
import time
from functools import partial

import numpy as np
from gm.data.loader import create_loader
from gm.helpers import (
    EMA,
    SD_XL_BASE_RATIOS,
    VERSION2SPECS,
    create_model,
    get_all_reduce_config,
    get_grad_reducer,
    get_learning_rate,
    get_loss_scaler,
    get_optimizer,
    load_checkpoint,
    pre_compile_graph,
    save_checkpoint,
    set_default,
)
from gm.util.util import auto_mixed_precision
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import Tensor, nn


def get_parser_train():
    parser = argparse.ArgumentParser(description="train with sd-xl")
    parser.add_argument("--version", type=str, default="SDXL-base-1.0", choices=["SDXL-base-1.0", "SDXL-refiner-1.0"])
    parser.add_argument("--config", type=str, default="configs/training/sd_xl_base_finetune_lora.yaml")
    parser.add_argument(
        "--task",
        type=str,
        default="txt2img",
        choices=["txt2img", "cache"],
    )
    parser.add_argument("--cache_latent", type=ast.literal_eval, default=False)
    parser.add_argument("--cache_text_embedding", type=ast.literal_eval, default=False)
    parser.add_argument("--cache_path", type=str, default="./cache_data")

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--clip_grad", default=False, type=ast.literal_eval, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--use_ema", action="store_true", help="whether use ema")
    parser.add_argument("--weight", type=str, default="checkpoints/sd_xl_base_1.0_ms.ckpt")
    parser.add_argument("--per_batch_size", type=int, default=None)
    parser.add_argument("--scale_lr", type=ast.literal_eval, default=False)
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default=None,
        choices=["earlier", "later", "range"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        default=None,
        type=float,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sd_xl_base_ratios", type=str, default="1.0")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="./runs")
    parser.add_argument("--save_path_with_time", type=ast.literal_eval, default=True)
    parser.add_argument("--log_interval", type=int, default=1, help="log interval")
    parser.add_argument("--save_ckpt_interval", type=int, default=1000, help="save ckpt interval")
    parser.add_argument(
        "--max_num_ckpt",
        type=int,
        default=None,
        help="Max number of ckpts saved. If exceeds, delete the oldest one. Set None: keep all ckpts.",
    )
    parser.add_argument("--resume_step", type=int, default=0, help="resume from step_n")
    parser.add_argument("--optimizer_weight", type=str, default=None, help="load optimizer weight")
    parser.add_argument("--save_optimizer", type=ast.literal_eval, default=False, help="enable save optimizer")
    parser.add_argument("--data_sink", type=ast.literal_eval, default=False)
    parser.add_argument("--sink_size", type=int, default=1000)
    parser.add_argument(
        "--dataset_load_tokenizer", type=ast.literal_eval, default=True, help="create dataset with tokenizer"
    )
    parser.add_argument("--lpw", type=ast.literal_eval, default=False)
    parser.add_argument("--max_embeddings_multiples", type=int, default=3, help="control the length of long prompts")

    # args for infer
    parser.add_argument("--infer_during_train", type=ast.literal_eval, default=False)
    parser.add_argument("--infer_interval", type=int, default=1, help="log interval")

    # args for env
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)"
    )
    parser.add_argument("--ms_amp_level", type=str, default="O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    parser.add_argument("--ms_enable_allreduce_fusion", type=ast.literal_eval, default=True)
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

    # args for dynamic shape
    parser.add_argument("--dynamic_shape", type=ast.literal_eval, default=False)
    return parser


def train(args):
    # 1. Init Env
    args = set_default(args)

    # 2. Create LDM Engine
    config = OmegaConf.load(args.config)
    model, _ = create_model(
        config,
        checkpoints=args.weight,
        freeze=False,
        load_filter=False,
        param_fp16=args.param_fp16,
        amp_level=args.ms_amp_level,
        load_first_stage_model=not args.cache_latent,
        load_conditioner=not args.cache_text_embedding,
    )
    if isinstance(model.model, nn.Cell):
        model.model.set_train(True)  # only unet

    # 3. Create dataloader
    assert "data" in config
    per_batch_size = config.data.pop("per_batch_size")
    per_batch_size = per_batch_size if args.per_batch_size is None else args.per_batch_size
    dataloader = create_loader(
        data_path=args.data_path,
        rank=args.rank,
        rank_size=args.rank_size,
        tokenizer=model.conditioner.tokenize
        if (args.dataset_load_tokenizer and not args.cache_text_embedding)
        else None,
        token_nums=len(model.conditioner.embedders)
        if (args.dataset_load_tokenizer and not args.cache_text_embedding)
        else None,
        cache_latent=args.cache_latent,
        cache_text_embedding=args.cache_text_embedding,
        cache_path=args.cache_path,
        per_batch_size=per_batch_size,
        lpw=args.lpw,
        max_embeddings_multiples=args.max_embeddings_multiples,
        **config.data,
    )
    total_step = config.data.total_step if hasattr(config.data, "total_step") else dataloader.get_dataset_size()
    random.seed(args.seed)  # for multi_aspect

    # 4. Create train step func
    assert "sigma_sampler_config" in config.model.params
    num_timesteps = config.model.params.sigma_sampler_config.params.num_idx
    timestep_bias_weighting = generate_timestep_weights(args, num_timesteps)

    assert "optim" in config
    scaler = args.rank_size * dataloader.get_batch_size() * args.gradient_accumulation_steps if args.scale_lr else 1.0
    lr = get_learning_rate(config.optim, total_step, scaler)
    if "scheduler_config" in config.optim and args.resume_step:
        lr = lr[args.resume_step :]

    scaler = get_loss_scaler(ms_loss_scaler="static", scale_value=1024)
    if args.ms_enable_allreduce_fusion and args.rank_size > 1:
        trainable_params, all_reduce_fusion_config = get_all_reduce_config(model)
        ms.set_auto_parallel_context(all_reduce_fusion_config=all_reduce_fusion_config)
    else:
        trainable_params = model.model.trainable_params()
        if model.conditioner is not None:
            trainable_params += model.conditioner.trainable_params()
    if isinstance(model.model, nn.Cell):
        optimizer = get_optimizer(config.optim, lr, params=trainable_params)
        reducer = get_grad_reducer(is_parallel=args.is_parallel, parameters=optimizer.parameters)
        if args.optimizer_weight:
            print(f"Loading optimizer from {args.optimizer_weight}")
            load_checkpoint(optimizer, args.optimizer_weight, remove_prefix="ldm_with_loss_grad.optimizer.")
    else:
        optimizer, reducer = None, None

    if args.use_ema:
        ema = EMA(model, ema_decay=0.9999)
    else:
        ema = None

    if args.ms_mode == 1:
        # Pynative Mode
        assert args.timestep_bias_strategy is None, "Not support timestep bias strategy."
        assert args.snr_gamma is None, "Not supports snr_gamma."
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
            from gm.models.trainer_factory import TrainOneStepCell

            # model = auto_mixed_precision(model, amp_level=args.ms_amp_level)
            model.model = auto_mixed_precision(model.model, amp_level=args.ms_amp_level)

            train_step_fn = TrainOneStepCell(
                model,
                optimizer,
                reducer,
                scaler,
                overflow_still_update=args.overflow_still_update,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                clip_grad=args.clip_grad,
                clip_norm=args.max_grad_norm,
                enable_first_stage_model=not args.cache_latent,
                enable_conditioner=not args.cache_text_embedding,
                ema=ema,
                timestep_bias_weighting=timestep_bias_weighting,
                snr_gamma=args.snr_gamma,
            )

            if args.dynamic_shape:
                input_dyn = Tensor(shape=[per_batch_size, 3, None, None], dtype=ms.float32)
                token1 = Tensor(np.ones((per_batch_size, 77)),dtype=ms.int32)
                token2 = Tensor(np.ones((per_batch_size, 77)),dtype=ms.int32)
                token3 = Tensor(np.ones((per_batch_size, 2)),dtype=ms.float32)
                token4 = Tensor(np.ones((per_batch_size, 2)),dtype=ms.float32)
                token5 = Tensor(np.ones((per_batch_size, 2)),dtype=ms.float32)
                token = [token1, token2, token3, token4, token5]

                train_step_fn.set_inputs(input_dyn, *token)

            if model.disable_first_stage_amp and train_step_fn.first_stage_model is not None:
                train_step_fn.first_stage_model.to_float(ms.float32)
            jit_config = ms.JitConfig()
        else:
            from gm.models.trainer_factory import TrainerMultiGraphTwoStage

            assert args.version == "SDXL-base-1.0", "Only supports sdxl-base."
            assert args.task == "txt2img", "Only supports text2img task."
            assert args.optimizer_weight is None, "Not supports load optimizer weight."
            assert args.timestep_bias_strategy is None, "Not support timestep bias strategy."
            assert args.snr_gamma is None, "Not supports snr_gamma."
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

    # 5. Start Training
    if args.max_num_ckpt is not None and args.max_num_ckpt <= 0:
        raise ValueError("args.max_num_ckpt must be None or a positive integer!")
    if args.task == "txt2img":
        train_fn = train_txt2img if not args.data_sink else train_txt2img_datasink
        train_fn(
            args, train_step_fn, dataloader=dataloader, optimizer=optimizer, model=model, jit_config=jit_config, ema=ema
        )
    elif args.task == "img2img":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown task {args.task}")


def train_txt2img(
    args, train_step_fn, dataloader, optimizer=None, model=None, ema=None, **kwargs
):  # for print  # for infer/ckpt
    dtype = ms.float32 if args.ms_amp_level not in ("O2", "O3") else ms.float16
    total_step = dataloader.get_dataset_size()
    loader = dataloader.create_tuple_iterator(output_numpy=True, num_epochs=1)

    # pre compile graph
    if args.lpw:
        pre_compile_graph(args.config, args.per_batch_size, train_step_fn, args.rank, args.max_embeddings_multiples)

    s_time = time.time()
    ckpt_queue = []
    for i, data in enumerate(loader):
        if i > total_step - args.resume_step:
            break
        i += args.resume_step
        if args.dataset_load_tokenizer or args.cache_text_embedding:
            image, tokens = data[0], data[1:]
            image, tokens = Tensor(image), [Tensor(t) for t in tokens]
        else:
            data = data[0]
            data = {k: (Tensor(v, dtype) if k != "txt" else v.tolist()) for k, v in data.items()}

            image = data[model.input_key]
            tokens, _ = model.conditioner.tokenize(
                data, lpw=args.lpw, max_embeddings_multiples=args.max_embeddings_multiples
            )
            tokens = [Tensor(t) for t in tokens]

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
                f"Step {i + 1}/{total_step}, size: {image.shape[:]}, lr: {cur_lr}, loss: {loss.asnumpy():.6f}"
                f", time cost: {(time.time()-s_time) * 1000 / args.log_interval:.2f} ms",
                flush=True,
            )
            s_time = time.time()

        # Save checkpoint
        if (i + 1) % args.save_ckpt_interval == 0 and args.rank % 8 == 0:
            save_ckpt_dir = os.path.join(args.save_path, "weights", args.version + f"_{(i + 1)}.ckpt")
            if args.cache_latent and args.cache_text_embedding:
                save_ckpt_dir = os.path.join(args.save_path, "weights", f"unet_{(i + 1)}.ckpt")

            if isinstance(model.model, nn.Cell):
                model.model.set_train(False)  # only unet
                save_checkpoint(
                    model if not ema else ema,
                    save_ckpt_dir,
                    ckpt_queue,
                    args.max_num_ckpt,
                    only_save_lora=False
                    if not hasattr(model.model.diffusion_model, "only_save_lora")
                    else model.model.diffusion_model.only_save_lora,
                )
                model.model.set_train(True)  # only unet
            else:
                model.save_checkpoint(save_ckpt_dir)
            ckpt_queue.append(save_ckpt_dir)

            if args.save_optimizer:
                save_optimizer_dir = os.path.join(args.save_path, "optimizer.ckpt")
                ms.save_checkpoint(optimizer, save_optimizer_dir)
                print(f"save optimizer weight to {save_optimizer_dir}")

        # Infer during train
        if (i + 1) % args.infer_interval == 0 and args.infer_during_train:
            print(f"Step {i + 1}/{total_step}, infer starting...")
            infer_during_train(
                model=model,
                prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                save_path=os.path.join(args.save_path, "txt2img/", f"step_{i+1}_rank_{args.rank}"),
            )
            print(f"Step {i + 1}/{total_step}, infer done.", flush=True)


def train_txt2img_datasink(
    args, train_step_fn, dataloader, optimizer=None, model=None, jit_config=None, ema=None, **kwargs
):  # for print  # for infer/ckpt
    total_step = dataloader.get_dataset_size()
    epochs = total_step // args.sink_size
    assert args.dataset_load_tokenizer

    train_fn_sink = ms.data_sink(fn=train_step_fn, dataset=dataloader, sink_size=args.sink_size, jit_config=jit_config)

    ckpt_queue = []
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
            if args.cache_latent and args.cache_text_embedding:
                save_ckpt_dir = os.path.join(args.save_path, "weights", f"unet_{cur_step}.ckpt")

            if isinstance(model.model, nn.Cell):
                model.model.set_train(False)  # only unet
                save_checkpoint(
                    model if not ema else ema,
                    save_ckpt_dir,
                    ckpt_queue,
                    args.max_num_ckpt,
                    only_save_lora=False
                    if not hasattr(model.model.diffusion_model, "only_save_lora")
                    else model.model.diffusion_model.only_save_lora,
                )
                model.model.set_train(True)  # only unet
            else:
                model.save_checkpoint(save_ckpt_dir)
            ckpt_queue.append(save_ckpt_dir)

        # Infer during train
        if cur_step % args.infer_interval == 0 and args.infer_during_train:
            print(f"Step {cur_step}/{total_step}, infer starting...")
            infer_during_train(
                model=model,
                prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                save_path=os.path.join(args.save_path, "txt2img/", f"step_{cur_step}_rank_{args.rank}"),
            )
            print(f"Step {cur_step}/{total_step}, infer done.", flush=True)


def infer_during_train(model, prompt, save_path, lpw=False):
    from gm.helpers import init_sampling, perform_save_locally

    version_dict = VERSION2SPECS.get(args.version)
    W, H = SD_XL_BASE_RATIOS[args.sd_xl_base_ratios]
    C = version_dict["C"]
    F = version_dict["f"]
    is_legacy = version_dict["is_legacy"]

    value_dict = {
        "prompt": prompt,
        "negative_prompt": "",
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
        "crop_coords_top": 0,
        "crop_coords_left": 0,
        "aesthetic_score": 6.0,
        "negative_aesthetic_score": 2.5,
    }
    sampler, num_rows, num_cols = init_sampling(steps=40, num_cols=1)

    out = model.do_sample(
        sampler,
        value_dict,
        num_rows * num_cols,
        H,
        W,
        C,
        F,
        force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        return_latents=False,
        filter=None,
        amp_level="O2",
        lpw=lpw,
    )
    perform_save_locally(save_path, out)


def cache_data(args):
    # 1. Init Env
    args = set_default(args)

    # 2. Create LDM Engine
    config = OmegaConf.load(args.config)
    model, _ = create_model(
        config,
        checkpoints=args.weight,
        freeze=False,
        load_filter=False,
        param_fp16=args.param_fp16,
        amp_level=args.ms_amp_level,
    )
    conditioner = model.conditioner
    first_stage_model = model.first_stage_model
    if model.disable_first_stage_amp:
        first_stage_model.to_float(ms.float32)

    # 3. Create Dataloader
    assert "data" in config
    config.data.pop("per_batch_size")
    config.data.pop("total_step")
    config.data.pop("shuffle")
    dataloader = create_loader(
        data_path=args.data_path,
        rank=args.rank,
        rank_size=args.rank_size,
        tokenizer=model.conditioner.tokenize if args.dataset_load_tokenizer else None,
        token_nums=len(model.conditioner.embedders) if args.dataset_load_tokenizer else None,
        per_batch_size=1,
        total_step=1,
        shuffle=False,
        return_sample_name=True,
        **config.data,
    )

    # 4. Cache Data
    os.makedirs(args.cache_path, exist_ok=False)
    if args.cache_latent:
        os.makedirs(os.path.join(args.cache_path, "latent_cache"), exist_ok=False)
    if args.cache_text_embedding:
        os.makedirs(os.path.join(args.cache_path, "vector_cache"), exist_ok=False)
        os.makedirs(os.path.join(args.cache_path, "crossattn_cache"), exist_ok=False)

    dtype = ms.float32 if args.ms_amp_level not in ("O2", "O3") else ms.float16
    total_num = dataloader.get_dataset_size()
    loader = dataloader.create_tuple_iterator(output_numpy=True, num_epochs=1)
    latent, vector, crossattn = None, None, None
    s_time = time.time()

    for i, data in enumerate(loader):
        if not args.dataset_load_tokenizer:
            # Get data, image and tokens, to tensor
            data = data[0]
            data = {k: (Tensor(v, dtype) if k not in ("txt", "sample_name") else v.tolist()) for k, v in data.items()}

            image = data[model.input_key]
            tokens, _ = model.conditioner.tokenize(data)
            tokens = [Tensor(t) for t in tokens]
            sample_name = data["sample_name"][0]
        else:
            image, tokens, sample_name = data[0], data[1:-1], data[-1][0]
            image, tokens = Tensor(image), [Tensor(t) for t in tokens]

        if args.cache_latent:
            latent = first_stage_model.encode(image)
            np.save(os.path.join(args.cache_path, "latent_cache", f"{sample_name}.npy"), latent.asnumpy())
        if args.cache_text_embedding:
            vector, crossattn, _ = conditioner(*tokens)
            np.save(os.path.join(args.cache_path, "vector_cache", f"{sample_name}.npy"), vector.asnumpy())
            np.save(os.path.join(args.cache_path, "crossattn_cache", f"{sample_name}.npy"), crossattn.asnumpy())

        # Print meg
        if (i + 1) % args.log_interval == 0:
            print(
                f"Rank {args.rank + 1}/{args.rank_size}, Cache sample {i + 1}/{total_num}, "
                f"Size of image/latent: "
                f"{image.shape[:]}/{latent.shape[:] if latent is not None else None}, "
                f"Size of vector/crossattn: "
                f"{vector.shape[:] if vector is not None else None}/"
                f"{crossattn.shape[:] if crossattn is not None else None}"
                f", time cost: {(time.time() - s_time) * 1000 / args.log_interval:.2f} ms",
                flush=True,
            )
            s_time = time.time()

    print(f"Rank {args.rank + 1}/{args.rank_size}, Cache sample {total_num}, Done.")


def generate_timestep_weights(args, num_timesteps):
    weights = np.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # None or any other string
        return None
    if args.timestep_bias_multiplier <= 0:
        raise ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return Tensor(weights, ms.float32)


if __name__ == "__main__":
    parser = get_parser_train()
    args, _ = parser.parse_known_args()
    if args.task == "cache":
        cache_data(args)
    else:
        train(args)
