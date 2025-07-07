# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import shutil
from pathlib import Path
from time import time
from typing import Any, Dict, Optional

import numpy as np
import yaml
from args import get_args
from cogvideox.dataset import VideoDatasetWithResizeAndRectangleCrop, VideoDatasetWithResizing
from cogvideox.datasets.bucket import Bucket, bucket_split_function
from cogvideox.models import AutoencoderKLCogVideoX_SP, CogVideoXTransformer3DModel_SP
from cogvideox.utils import get_optimizer
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import mindspore as ms
import mindspore.dataset as ds
from mindspore import mint, nn, ops
from mindspore.amp import auto_mixed_precision
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

from mindone.diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, ConfigMixin, SchedulerMixin
from mindone.diffusers.optimization import get_scheduler
from mindone.diffusers.training_utils import (
    AttrJitWrapper,
    cast_training_params,
    init_distributed_device,
    is_local_master,
    is_master,
    prepare_train_network,
    pynative_no_grad,
    set_seed,
)
from mindone.diffusers.utils import pynative_context
from mindone.diffusers.utils.logging import get_logger
from mindone.diffusers.utils.mindspore_utils import get_state_dict
from mindone.transformers import T5EncoderModel

logger = get_logger(__name__)


class CollateFunction:
    def __init__(self, use_rope: bool) -> None:
        self.use_rope = use_rope

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, ms.Tensor]:
        text_input_ids = [x["text_input_ids"] for x in examples]
        text_input_ids = np.stack(text_input_ids)

        videos = [x["video"] for x in examples]
        videos = np.stack(videos)

        if self.use_rope:
            rotary_positional_embeddings = [x["rotary_positional_embeddings"] for x in examples]
            rotary_positional_embeddings = np.stack(rotary_positional_embeddings)
            return videos, text_input_ids, rotary_positional_embeddings
        else:
            return videos, text_input_ids


class FilterFunction:
    def __init__(self, use_rope: bool) -> None:
        self.use_rope = use_rope

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, ms.Tensor]:
        if self.use_rope:
            return examples["video"], examples["text_input_ids"], examples["rotary_positional_embeddings"]
        else:
            return examples["video"], examples["text_input_ids"]


def set_params_requires_grad(m: nn.Cell, requires_grad: bool):
    for p in m.get_parameters():
        p.requires_grad = requires_grad


def main(args):
    # If passed along, set the training seed now.
    if args.seed is not None:
        args.seed = 1234
    set_seed(args.seed)
    # Init context about MindSpore
    ms.set_context(
        mode=args.mindspore_mode,
        jit_config={"jit_level": args.jit_level},
    )
    ms.set_cpu_affinity(True)

    # read attr distributed, writer attrs rank/local_rank/world_size:
    #   args.local_rank = mindspore.communication.get_local_rank()
    #   args.world_size = mindspore.communication.get_group_size()
    #   args.rank = mindspore.communication.get_rank()
    init_distributed_device(args)

    logging_dir = Path(args.output_dir, args.logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the repository creation
    if is_master(args):
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)

    # enable_sequence_parallelism check
    enable_sequence_parallelism = getattr(args, "enable_sequence_parallelism", False)
    sequence_parallel_shards = getattr(args, "sequence_parallel_shards", 1)
    if enable_sequence_parallelism:
        if args.world_size <= 1 or args.sequence_parallel_shards <= 1:
            logging.warning(
                f"world_size :{args.world_size}, "
                f"sequence_parallel_shards: {args.sequence_parallel_shards} "
                f"can not enable enable_sequence_parallelism=True."
            )
            enable_sequence_parallelism = False
            sequence_parallel_shards = 1
        else:
            from cogvideox.acceleration import create_parallel_group

            create_parallel_group(sequence_parallel_shards=args.sequence_parallel_shards)
            ms.set_auto_parallel_context(enable_alltoall=True)
    else:
        sequence_parallel_shards = 1

    # Prepare models and scheduler
    # Loading order changed for MindSpore adaptation
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = ms.float32
    if args.mixed_precision == "fp16":
        weight_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = ms.bfloat16

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    # load_dtype = ms.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else ms.float16
    if args.transformer_config is None:
        transformer = CogVideoXTransformer3DModel_SP.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            mindspore_dtype=weight_dtype,
            revision=args.revision,
            variant=args.variant,
            max_text_seq_length=args.max_sequence_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
    elif os.path.exists(args.transformer_config):
        with open(args.transformer_config) as f:
            config = yaml.safe_load(f)["transformer"]
            config["max_text_seq_length"] = args.max_sequence_length
            config["enable_sequence_parallelism"] = enable_sequence_parallelism
            transformer = CogVideoXTransformer3DModel_SP(**config)
            logger.info(f"Build transformer model from {args.transformer_config}")
            if os.path.exists(args.transformer_ckpt_path):
                ms.load_checkpoint(args.transformer_ckpt_path, transformer)
                logger.info(f"Load transformer checkpoint from {args.transformer_ckpt_path}")

    else:
        raise ValueError(f"transformer_config: {args.transformer_config} is not exist!")

    transformer.fa_checkpointing = args.fa_gradient_checkpointing

    text_encoder, vae = None, None
    # Only load Text-encoder & VAE when they are needed in training. Because currently
    # the MindSpore memory pool does not have the function of releasing memory fragments. Deleting
    # them after loading still causes memory fragments which might result in OOM on devices.
    if not args.embeddings_cache:
        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            mindspore_dtype=weight_dtype,
            revision=args.revision,
        )
        set_params_requires_grad(text_encoder, False)

    if not args.vae_cache:
        if enable_sequence_parallelism:
            vae = AutoencoderKLCogVideoX_SP.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="vae",
                mindspore_dtype=weight_dtype,
                revision=args.revision,
                variant=args.variant,
            )
        else:
            vae = AutoencoderKLCogVideoX.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="vae",
                mindspore_dtype=weight_dtype,
                revision=args.revision,
                variant=args.variant,
            )
        for param in vae.get_parameters():
            param.parallel_optimizer = False

        if args.enable_slicing:
            vae.enable_slicing()
        if args.enable_tiling:
            vae.enable_tiling()

        set_params_requires_grad(vae, False)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, output_dir):
        if args.zero_stage == 3:
            all_gather_op = ops.AllGather()
        for model in models:
            if isinstance(model, CogVideoXTransformer3DModel_SP):
                params = []
                for param in model.get_parameters(expand=True):
                    if args.zero_stage == 3 and param.parallel_optimizer:
                        data = ms.Tensor(all_gather_op(param).asnumpy())
                    else:
                        data = ms.Tensor(param.asnumpy())
                    params.append({"name": param.name, "data": data})
                if is_local_master(args):
                    ms.save_checkpoint(params, os.path.join(output_dir, "transformer"))
            else:
                print(f"Unexpected save model: {model.__class__}")

    def load_model_hook(models, input_dir):
        for model in models:
            if isinstance(model, type(transformer)):
                transformer_ = model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")

        load_model = CogVideoXTransformer3DModel_SP.from_pretrained(
            os.path.join(input_dir, "transformer"), mindspore_dtype=weight_dtype
        )
        transformer_.register_to_config(**load_model.config)
        ms.load_param_into_net(transformer_, get_state_dict(load_model, name_prefix="transformer"))
        del load_model

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    # filter unnecessary optimizer state for loading & saving state
    def optimizer_state_filter(param_name: str):
        # Saving optimizer directly will save all trainable parameters of transformer model besides
        # optimizer states, we set `choice_func` of mindspore.save_checkpoint(optimizer) to `optimizer_state_filter`
        # to NOT save & load these parameters since they will be updated automatically when `load_model_hook`
        # and are unnecessary to save and load. This saves a lot of disk space.
        return not param_name.startswith("transformer.")

    # Define models to load or save for load_model_hook() and save_model_hook()
    models = [transformer]

    # Dataset and DataLoader
    # Moving tokenize & prepare RoPE in dataset preprocess, which need some configs
    transformer_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    vae_config = AutoencoderKLCogVideoX.from_config(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )

    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae_config.config.block_out_channels) - 1)

    all_buckets = None
    if not args.vae_cache and args.bucket_config:
        with open(args.bucket_config, "r") as f:
            config = yaml.safe_load(f)
            all_buckets = Bucket(config["bucket_config"])
            print("enable bucketed training")

    dataset_init_kwargs = {
        "data_root": args.data_root,
        "dataset_file": args.dataset_file,
        "caption_column": args.caption_column,
        "video_column": args.video_column,
        "max_num_frames": args.max_num_frames,
        "id_token": args.id_token,
        "height_buckets": args.height_buckets,
        "width_buckets": args.width_buckets,
        "frame_buckets": args.frame_buckets,
        "embeddings_cache": args.embeddings_cache,
        "vae_cache": args.vae_cache,
        "random_flip": args.random_flip,
        "tokenizer": None if args.embeddings_cache else tokenizer,
        "max_sequence_length": None if args.embeddings_cache else args.max_sequence_length,
        "use_rotary_positional_embeddings": transformer_config.use_rotary_positional_embeddings,
        "vae_scale_factor_spatial": VAE_SCALE_FACTOR_SPATIAL,
        "patch_size": transformer_config.patch_size,
        "patch_size_t": transformer_config.patch_size_t if hasattr(transformer_config, "patch_size_t") else None,
        "attention_head_dim": transformer_config.attention_head_dim,
        "base_height": transformer_config.sample_height * VAE_SCALE_FACTOR_SPATIAL,
        "base_width": transformer_config.sample_width * VAE_SCALE_FACTOR_SPATIAL,
        "buckets": all_buckets,
    }
    if args.video_reshape_mode is None:
        train_dataset = VideoDatasetWithResizing(**dataset_init_kwargs)
    else:
        train_dataset = VideoDatasetWithResizeAndRectangleCrop(
            video_reshape_mode=args.video_reshape_mode, **dataset_init_kwargs
        )

    if args.enable_sequence_parallelism:
        data_device_num = args.world_size // args.sequence_parallel_shards
        data_rank_id = args.rank // args.sequence_parallel_shards
        logger.info(f"Creating dataloader: ID={args.rank}, group={data_rank_id}, num_groups={data_device_num}")
    else:
        data_device_num = args.world_size
        data_rank_id = args.rank

    ds.config.set_prefetch_size(4)
    # ds.config.set_iterator_mode(do_copy=True, parallel_convert=True)
    train_dataloader = ds.GeneratorDataset(
        train_dataset,
        column_names=["examples"],
        shard_id=data_rank_id,
        num_shards=data_device_num,
        num_parallel_workers=args.dataloader_num_workers,
    )
    if all_buckets:
        hash_func, bucket_boundaries, bucket_batch_sizes = bucket_split_function(all_buckets)
        filter_fn = FilterFunction(transformer_config.use_rotary_positional_embeddings)
        train_dataloader = train_dataloader.map(
            filter_fn,
            input_columns=["examples"],
            output_columns=["videos", "text_input_ids", "rotary_positional_embeddings"],
        )

        train_dataloader = train_dataloader.bucket_batch_by_length(
            ["videos"],
            bucket_boundaries,
            bucket_batch_sizes,
            element_length_function=hash_func,
            drop_remainder=False,
        )
    else:
        collate_fn = CollateFunction(transformer_config.use_rotary_positional_embeddings)
        train_dataloader = train_dataloader.batch(
            batch_size=args.train_batch_size,
            per_batch_map=lambda examples, batch_info: collate_fn(examples),
            input_columns=["examples"],
            output_columns=["videos", "text_input_ids", "rotary_positional_embeddings"]
            if transformer_config.use_rotary_positional_embeddings
            else ["videos", "text_input_ids"],
            num_parallel_workers=args.dataloader_num_workers,
        )
    data_len = len(train_dataloader)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(data_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * args.world_size
            / sequence_parallel_shards
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Make sure the trainable params are in float32.
    # Do AMP wrapper manually
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=ms.float32)
        transformer = auto_mixed_precision(transformer, amp_level=args.amp_level, dtype=ms.float16)

    # Optimization parameters
    # Do Not define grouped learning rate here since it is not used but results in Call Depth Overflow failure
    # It might be a design flaw of MindSpore optimizer which should be fixed, but now we just avoid it.
    transformer_parameters = transformer.trainable_params()
    num_trainable_parameters = sum(param.numel() for param in transformer_parameters)

    # Bias and time_embedding does not set optimizer parallelism
    for param in transformer_parameters:
        if "bias" in param.name or "time_embedding" in param.name:
            param.parallel_optimizer = False

    optimizer = get_optimizer(
        params_to_optimize=transformer_parameters,
        optimizer_name=args.optimizer,
        learning_rate=lr_scheduler,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(data_len / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # create train_step for training
    train_step = TrainStepForCogVideo(
        vae=vae,
        vae_config=vae_config,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        weight_dtype=weight_dtype,
        args=args,
        use_rotary_positional_embeddings=transformer_config.use_rotary_positional_embeddings,
        enable_sequence_parallelism=enable_sequence_parallelism,
    ).set_train(True)

    loss_scaler = DynamicLossScaleUpdateCell(loss_scale_value=65536.0, scale_factor=2, scale_window=2000)
    train_step = prepare_train_network(
        train_step,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=True,
        clip_norm=args.max_grad_norm,
        zero_stage=args.zero_stage,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if is_master(args):
        with open(logging_dir / "hparams.yml", "w") as f:
            yaml.dump(vars(args), f, indent=4)
    trackers = dict()
    for tracker_name in args.report_to.split(","):
        if tracker_name == "tensorboard":
            trackers[tracker_name] = SummaryWriter(str(logging_dir), write_to_disk=is_master(args))
        else:
            logger.warning(f"Tracker {tracker_name} is not implemented, omitting...")

    tracker_name = args.tracker_name or "cogvideox-sft"

    # Train!
    total_batch_size = args.train_batch_size * args.world_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters/1e9:.2f} B")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {data_len}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            if is_master(args):
                logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            if is_master(args):
                logger.info(f"Resuming from {path}")
            # TODO: load optimizer & grad scaler etc. like accelerator.load_state
            load_model_hook(models, os.path.join(args.output_dir, path))
            train_step.load_state(args, os.path.join(args.output_dir, path), optimizer_state_filter)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            lr_scheduler = lr_scheduler[initial_global_step:]

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not is_master(args),
    )

    train_dataloader_iter = train_dataloader.create_tuple_iterator(num_epochs=args.num_train_epochs - first_epoch)

    if args.mindspore_mode == ms.GRAPH_MODE and args.dynamic_shape:
        symbol_batch_size = ms.Symbol(unique=True)
        symbol_frames = ms.Symbol(unique=True)
        symbol_height = ms.Symbol(unique=True)
        symbol_width = ms.Symbol(unique=True)
        symbol_sequence_length = ms.Symbol(unique=True)

        channels = vae_config.config.norm_num_groups
        videos_shape = (symbol_batch_size, symbol_frames, channels // 2, symbol_height, symbol_width)
        dummy_videos = ms.Tensor(shape=videos_shape, dtype=weight_dtype)

        num_tokens = args.max_sequence_length
        text_encoder_config_path = Path(args.pretrained_model_name_or_path) / "text_encoder" / "config.json"
        text_encoder_config = json.load(open(text_encoder_config_path))
        hidden_size = text_encoder_config["d_model"]
        prompt_embeds_shape = (symbol_batch_size, num_tokens, hidden_size)
        dummy_prompt_embeds = ms.Tensor(shape=prompt_embeds_shape, dtype=weight_dtype)
        dummy_timesteps = ms.Tensor(shape=(symbol_batch_size,), dtype=ms.int32)
        cos_and_sin_dim = 2
        dim = dataset_init_kwargs["attention_head_dim"]
        rotary_positional_embeddings_shape = (cos_and_sin_dim, symbol_sequence_length, dim)
        dummy_rotary_positional_embeddings = ms.Tensor(shape=rotary_positional_embeddings_shape, dtype=weight_dtype)

        train_step.set_inputs(
            dummy_videos, dummy_prompt_embeds, dummy_videos, dummy_timesteps, dummy_rotary_positional_embeddings
        )

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.set_train(True)

        for step, batch in enumerate(train_dataloader_iter):
            train_start = time()
            # videos B, C, F, H, W: (1, 32, 20, 96, 170)
            transformer_inputs = train_step.network.prepare_transformer_inputs(*batch)
            loss, overflow, scale_sense = train_step(*transformer_inputs)
            if overflow:
                logger.warning(f"Step {step} overflow!, scale_sense is {scale_sense}")
            last_lr = optimizer.get_lr()
            last_lr = last_lr[0] if isinstance(last_lr, tuple) else last_lr  # grouped lr scenario
            logs = {"loss": loss.item(), "lr": last_lr.item(), "pre_step_time": time() - train_start}
            if args.dynamic_shape:
                logs["video_shape"] = batch[0].shape
            progress_bar.set_postfix(**logs)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if train_step.accum_steps == 1 or train_step.cur_accum_step.item() == 0:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if is_local_master(args):
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    # TODO: save optimizer & grad scaler etc. like accelerator.save_state
                    os.makedirs(save_path, exist_ok=True)
                    save_model_hook(models, save_path)
                    train_step.save_state(args, save_path, optimizer_state_filter)
                    logger.info(f"Saved state to {save_path}")

            if is_master(args):
                if args.dynamic_shape:
                    logs.pop("video_shape")
                for tracker_name, tracker in trackers.items():
                    if tracker_name == "tensorboard":
                        tracker.add_scalars("train", logs, global_step)

            if global_step >= args.max_train_steps:
                break


class TrainStepForCogVideo(nn.Cell):
    def __init__(
        self,
        vae: Optional[nn.Cell],
        vae_config: Optional[ConfigMixin],
        text_encoder: Optional[nn.Cell],
        transformer: nn.Cell,
        scheduler: SchedulerMixin,
        weight_dtype: ms.Type,
        args: AttrJitWrapper,
        use_rotary_positional_embeddings: bool,
        enable_sequence_parallelism: bool = False,
    ):
        super().__init__()

        vae_config = vae_config or vae.config

        self.weight_dtype = weight_dtype
        self.vae = vae
        self.vae_dtype = None if vae is None else vae.dtype
        self.vae_scaling_factor = vae_config.config.scaling_factor
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.scheduler = scheduler
        self.scheduler_num_train_timesteps = scheduler.config.num_train_timesteps

        self.use_rotary_positional_embeddings = use_rotary_positional_embeddings
        self.args = AttrJitWrapper(**vars(args))
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.sp_group = None
        self.sp_rank = 0
        if self.enable_sequence_parallelism:
            from cogvideox.acceleration import get_sequence_parallel_group

            from mindspore.communication import get_group_size, get_rank

            self.sp_group = get_sequence_parallel_group()
            self.global_rank = get_rank()
            self.sp_rank = get_rank(self.sp_group)
            self.sp_group_size = get_group_size(self.sp_group)

    def compute_prompt_embeddings(
        self,
        text_input_ids,
        num_videos_per_prompt: int = 1,
        dtype: ms.Type = None,
    ):
        batch_size = text_input_ids.shape[0]
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.tile((1, num_videos_per_prompt, 1))
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def diagonal_gaussian_distribution_sample(self, latent_dist: ms.Tensor) -> ms.Tensor:
        mean, logvar = mint.chunk(latent_dist, 2, dim=1)
        logvar = mint.clamp(logvar, min=-30.0, max=20.0)
        std = mint.exp(0.5 * logvar)

        sample = mint.randn_like(mean, dtype=mean.dtype)
        x = mean + std * sample
        return x

    def _broadcast_out(self, x):
        x = x.contiguous()
        if self.enable_sequence_parallelism:
            src_global_rank = self.global_rank // self.sp_group_size * self.sp_group_size
            mint.distributed.broadcast(tensor=x, src=src_global_rank, group=self.sp_group)
        return x

    def prepare_transformer_inputs(self, videos, text_input_ids_or_prompt_embeds, image_rotary_emb=None):
        with pynative_context(), pynative_no_grad():
            if not args.vae_cache:
                videos = videos.permute(0, 2, 1, 3, 4).to(self.vae.dtype)  # [B, C, F, H, W]
                videos = self.vae.encode(videos)[0]
            videos = videos.to(self.weight_dtype)
            videos = self.diagonal_gaussian_distribution_sample(videos) * self.vae_scaling_factor
            # [B, C, F, H, W] (1, 16, 20, 96, 170) -> [B, F, C, H, W]
            videos = videos.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            model_input = videos

            # Encode prompts
            if not args.embeddings_cache:
                prompt_embeds = self.compute_prompt_embeddings(
                    text_input_ids_or_prompt_embeds,
                    dtype=self.weight_dtype,
                )
            else:
                prompt_embeds = text_input_ids_or_prompt_embeds.to(dtype=self.weight_dtype)
            # prompt_embeds(1, 226, 4096)
            # Sample noise that will be added to the latents
            noise = mint.randn_like(model_input, dtype=model_input.dtype)
            batch_size, num_frames, num_channels, height, width = model_input.shape

            # Sample a random timestep for each image
            timesteps = mint.randint(0, self.scheduler_num_train_timesteps, (batch_size,), dtype=ms.int32)

            # Rotary embeds is Prepared in dataset.
            if self.use_rotary_positional_embeddings:
                image_rotary_emb = image_rotary_emb[0].to(self.weight_dtype)  # [2, S, D], (2, 40800, 64)

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = self.scheduler.add_noise(model_input, noise, timesteps)

            model_input = self._broadcast_out(model_input)
            prompt_embeds = self._broadcast_out(prompt_embeds)
            noisy_model_input = self._broadcast_out(noisy_model_input)
            timesteps = self._broadcast_out(timesteps)
            image_rotary_emb = self._broadcast_out(image_rotary_emb)
        return model_input, prompt_embeds, noisy_model_input, timesteps, image_rotary_emb

    def construct(self, model_input, prompt_embeds, noisy_model_input, timesteps, image_rotary_emb=None):
        batch_size = model_input.shape[0]

        # Predict the noise residual
        model_output = self.transformer(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        model_pred = self.scheduler.get_velocity(model_output, noisy_model_input, timesteps)

        alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=ms.float32)
        weights = 1 / (1 - alphas_cumprod[timesteps])
        new_weights_shape = weights.shape + (1,) * (len(model_pred.shape) - len(weights.shape))
        weights = weights.reshape(new_weights_shape)

        target = model_input

        loss = mint.mean(
            (weights * (model_pred - target) ** 2).reshape(batch_size, -1),
            dim=1,
        )
        loss = loss.mean()

        return loss


if __name__ == "__main__":
    args = get_args()
    main(args)
