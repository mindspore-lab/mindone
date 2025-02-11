import gc
import importlib
import json
import random
import sys
import time

import numpy as np
from hydit.config import get_args
from hydit.constants import T5_ENCODER, TEXT_ENCODER, TOKENIZER, VAE_EMA_PATH
from hydit.data_loader.arrow_load_stream import TextImageArrowStream
from hydit.diffusion import create_diffusion
from hydit.ds_config import deepspeed_config_from_args
from hydit.modules.fp16_layers import Float16Module
from hydit.modules.models import HUNYUAN_DIT_MODELS
from hydit.modules.posemb_layers import init_image_posemb
from hydit.modules.text_encoder import MT5Embedder
from hydit.utils.tools import create_exp_folder, get_trainable_params, model_resume
from IndexKits.index_kits import ResolutionGroup
from IndexKits.index_kits.sampler import BlockDistributedSampler, DistributedSamplerWithStartIndex
from transformers import BertTokenizer
from transformers import logging as tf_logging

import mindspore as ms
from mindspore import nn, ops
from mindspore.amp import DynamicLossScaler
from mindspore.communication import get_group_size, get_rank, init
from mindspore.dataset import GeneratorDataset

from mindone.diffusers._peft import LoraConfig, get_peft_model
from mindone.diffusers.models import AutoencoderKL
from mindone.diffusers.training_utils import TrainStep
from mindone.transformers import BertModel


def initialize(args, logger, model, opt, deepspeed_config):
    logger.info("Initialize deepspeed...")
    logger.info("    Using deepspeed optimizer")

    logger.info(
        f"    Building scheduler with warmup_min_lr={args.warmup_min_lr}, warmup_num_steps={args.warmup_num_steps}"
    )

    # model_parameters = get_trainable_params(model)
    model_parameters = model.trainable_params()
    opt_type = deepspeed_config["optimizer"]["type"]
    opt_params = deepspeed_config["optimizer"]["params"]
    opt = getattr(importlib.import_module("mindone.trainers.adamw_mint"), opt_type)(model_parameters, **opt_params)

    return model, opt


def save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir, by="step"):
    def save_lora_weight(checkpoint_dir, client_state, tag=f"{train_steps:07d}.ckpt"):
        cur_ckpt_save_dir = f"{checkpoint_dir}/{tag}"
        if rank == 0:
            model.module.save_pretrained(cur_ckpt_save_dir)

    def save_model_weight(client_state, tag):
        checkpoint_path = f"{checkpoint_dir}/{tag}"
        try:
            if args.training_parts == "lora":
                save_lora_weight(checkpoint_dir, client_state, tag=tag)
            else:
                ms.save_checkpoint(model, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Saved failed to {checkpoint_path}. {type(e)}: {e}")
            return False, ""
        return True, checkpoint_path

    client_state = {"steps": train_steps, "epoch": epoch, "args": args}
    # if ema is not None:
    #     client_state["ema"] = ema.state_dict()

    # Save model weights by epoch or step
    dst_paths = []
    if by == "epoch":
        tag = f"e{epoch:04d}.ckpt"
        dst_paths.append(save_model_weight(client_state, tag))
    elif by == "step":
        if train_steps % args.ckpt_every == 0:
            tag = f"{train_steps:07d}.ckpt"
            dst_paths.append(save_model_weight(client_state, tag))
        if train_steps % args.ckpt_latest_every == 0 or train_steps == args.max_training_steps:
            tag = "latest.ckpt"
            dst_paths.append(save_model_weight(client_state, tag))
    elif by == "final":
        tag = "final.ckpt"
        dst_paths.append(save_model_weight(client_state, tag))
    else:
        raise ValueError(f"Unknown save checkpoint method: {by}")

    saved = any([state for state, _ in dst_paths])
    if not saved:
        return False

    return True


def prepare_model_inputs(args, batch, vae, text_encoder, text_encoder_t5, freqs_cis_img):
    image, text_embedding, text_embedding_mask, text_embedding_t5, text_embedding_mask_t5, kwargs = batch

    # clip & mT5 text embedding
    encoder_hidden_states = text_encoder(
        ms.tensor(text_embedding),
        attention_mask=ms.tensor(text_embedding_mask),
    )[0]
    text_embedding_t5 = ms.tensor(text_embedding_t5).squeeze(1)
    text_embedding_mask_t5 = ms.tensor(text_embedding_mask_t5).squeeze(1)
    output_t5 = text_encoder_t5(
        input_ids=text_embedding_t5,
        attention_mask=text_embedding_mask_t5 if T5_ENCODER["attention_mask"] else None,
        output_hidden_states=True,
    )
    encoder_hidden_states_t5 = ops.stop_gradient(output_t5[1][T5_ENCODER["layer_index"]])

    # additional condition
    if args.size_cond:
        image_meta_size = kwargs["image_meta_size"]
    else:
        image_meta_size = None
    if args.use_style_cond:
        style = kwargs["style"]
    else:
        style = None

    if args.extra_fp16:
        image = ms.tensor(image).half()

    # Map input images to latent space + normalize latents:
    vae_scaling_factor = vae.config.scaling_factor
    latents = vae.diag_gauss_dist.sample(vae.encode(image)[0]).mul(vae_scaling_factor)

    # positional embedding
    _, _, height, width = image.shape
    reso = f"{height}x{width}"
    cos_cis_img, sin_cis_img = freqs_cis_img[reso]

    # Model conditions
    model_kwargs = dict(
        encoder_hidden_states=encoder_hidden_states,
        text_embedding_mask=text_embedding_mask,
        encoder_hidden_states_t5=encoder_hidden_states_t5,
        text_embedding_mask_t5=text_embedding_mask_t5,
        cos_cis_img=cos_cis_img,
        sin_cis_img=sin_cis_img,
    )
    if image_meta_size:
        model_kwargs["image_meta_size"] = image_meta_size
    if style:
        model_kwargs["style"] = style

    return latents, model_kwargs


def main(args):
    if args.training_parts == "lora":
        args.use_ema = False

    rank = 0
    world_size = 1
    batch_size = args.batch_size
    grad_accu_steps = args.grad_accu_steps
    global_batch_size = world_size * batch_size * grad_accu_steps

    ms.set_context(
        mode=0,
        device_target="Ascend",
        pynative_synchronize=True,
        jit_syntax_level=ms.STRICT,
        jit_config={"jit_level": "O1"},
    )

    if args.distributed:
        init()
        world_size = get_group_size()
        rank = get_rank()
        comm_fusion_dict = {"allreduce": {"mode": "auto", "config": None}}
        ms.context.set_auto_parallel_context(
            device_num=world_size,
            global_rank=rank,
            parallel_mode="data_parallel",
            gradients_mean=True,
            comm_fusion=comm_fusion_dict,
        )

    seed = args.global_seed * world_size + rank
    ms.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    deepspeed_config = deepspeed_config_from_args(args, global_batch_size)
    loss_scaler_params = deepspeed_config["loss_scaler"]
    gradient_clipping = deepspeed_config["gradient_clipping"]

    # Setup an experiment folder
    experiment_dir, checkpoint_dir, logger = create_exp_folder(args, rank)

    # Log all the arguments
    logger.info(sys.argv)
    logger.info(str(args))
    # Save to a json file
    args_dict = vars(args)
    args_dict["world_size"] = world_size
    with open(f"{experiment_dir}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    # Disable the message "Some weights of the model checkpoint at ... were not used when initializing BertModel."
    # If needed, just comment the following line.
    tf_logging.set_verbosity_error()

    # ===========================================================================
    # Building HYDIT
    # ===========================================================================

    logger.info("Building HYDIT Model.")

    # ---------------------------------------------------------------------------
    #   Training sample base size, such as 256/512/1024. Notice that this size is
    #   just a base size, not necessary the actual size of training samples. Actual
    #   size of the training samples are correlated with `resolutions` when enabling
    #   multi-resolution training.
    # ---------------------------------------------------------------------------
    image_size = args.image_size
    if len(image_size) == 1:
        image_size = [image_size[0], image_size[0]]
    if len(image_size) != 2:
        raise ValueError(f"Invalid image size: {args.image_size}")
    assert image_size[0] % 8 == 0 and image_size[1] % 8 == 0, (
        "Image size must be divisible by 8 (for the VAE encoder). " f"got {image_size}"
    )
    latent_size = [image_size[0] // 8, image_size[1] // 8]

    model = HUNYUAN_DIT_MODELS[args.model](
        args,
        input_size=latent_size,
        log_fn=logger.info,
    )
    # Multi-resolution / Single-resolution training.
    if args.multireso:
        resolutions = ResolutionGroup(
            image_size[0], align=16, step=args.reso_step, target_ratios=args.target_ratios
        ).data
    else:
        resolutions = ResolutionGroup(image_size[0], align=16, target_ratios=["1:1"]).data

    freqs_cis_img = init_image_posemb(
        args.rope_img,
        resolutions=resolutions,
        patch_size=model.patch_size,
        hidden_size=model.hidden_size,
        num_heads=model.num_heads,
        log_fn=logger.info,
        rope_real=args.rope_real,
    )

    # Create EMA model and convert to fp16 if needed.
    ema = None
    if args.use_ema:
        raise NotImplementedError("EMA feature is not supported.")

    # Setup gradient checkpointing
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Setup FP16 main model:
    if args.use_fp16:
        model = Float16Module(model, args)

    logger.info(f"    Using main model with data type {'fp16' if args.use_fp16 else 'fp32'}")

    # Setup VAE
    logger.info(f"    Loading vae from {VAE_EMA_PATH}")
    vae = AutoencoderKL.from_pretrained(VAE_EMA_PATH)
    # Setup BERT text encoder
    logger.info(f"    Loading Bert text encoder from {TEXT_ENCODER}")
    text_encoder = BertModel.from_pretrained(TEXT_ENCODER, False, revision=None)
    # Setup BERT tokenizer:
    logger.info(f"    Loading Bert tokenizer from {TOKENIZER}")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER)
    # Setup T5 text encoder
    mt5_path = T5_ENCODER["MT5"]
    embedder_t5 = MT5Embedder(mt5_path, mindspore_dtype=T5_ENCODER["mindspore_dtype"], max_length=args.text_len_t5)
    tokenizer_t5 = embedder_t5.tokenizer
    text_encoder_t5 = embedder_t5.model

    if args.extra_fp16:
        logger.info("    Using fp16 for extra modules: vae, text_encoder")
        vae = vae.half()
        text_encoder = text_encoder.half()
        text_encoder_t5 = text_encoder_t5.half()
    else:
        vae = vae
        text_encoder = text_encoder
        text_encoder_t5 = text_encoder_t5

    logger.info(f"    Optimizer parameters: lr={args.lr}, weight_decay={args.weight_decay}")
    logger.info("    Using deepspeed optimizer")
    opt = None

    # ===========================================================================
    # Building Dataset
    # ===========================================================================

    logger.info("Building Streaming Dataset.")
    logger.info(f"    Loading index file {args.index_file} (v2)")

    dataset = TextImageArrowStream(
        args=args,
        resolution=image_size[0],
        random_flip=args.random_flip,
        log_fn=logger.info,
        index_file=args.index_file,
        multireso=args.multireso,
        batch_size=batch_size,
        world_size=world_size,
        random_shrink_size_cond=args.random_shrink_size_cond,
        merge_src_cond=args.merge_src_cond,
        uncond_p=args.uncond_p,
        text_ctx_len=args.text_len,
        tokenizer=tokenizer,
        uncond_p_t5=args.uncond_p_t5,
        text_ctx_len_t5=args.text_len_t5,
        tokenizer_t5=tokenizer_t5,
    )
    if args.multireso:
        sampler = BlockDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=args.global_seed,
            shuffle=False,
            drop_last=True,
            batch_size=batch_size,
        )
    else:
        sampler = DistributedSamplerWithStartIndex(
            dataset, num_replicas=world_size, rank=rank, seed=args.global_seed, shuffle=False, drop_last=True
        )
    loader = GeneratorDataset(
        dataset,
        column_names=[
            "image",
            "text_embedding",
            "text_embedding_mask",
            "text_embedding_t5",
            "text_embedding_mask_t5",
            "kwargs",
        ],
        sampler=sampler,
        num_parallel_workers=args.num_workers,
        max_rowsize=-1,
    ).batch(batch_size=batch_size, drop_remainder=True, num_parallel_workers=args.num_workers)

    logger.info(f"    Dataset contains {len(dataset):,} images.")
    logger.info(f"    Index file: {args.index_file}.")
    if args.multireso:
        logger.info(
            f"    Using MultiResolutionBucketIndexV2 with step {dataset.index_manager.step} "
            f"and base size {dataset.index_manager.base_size}"
        )
        logger.info(f"\n  {dataset.index_manager.resolutions}")

    # ===========================================================================
    # Loading parameter
    # ===========================================================================

    logger.info("Loading parameter")
    start_epoch = 0
    start_epoch_step = 0
    train_steps = 0
    # Resume checkpoint if needed
    if args.resume:
        model, ema, start_epoch, start_epoch_step, train_steps = model_resume(args, model, ema, logger, len(loader))

    if args.training_parts == "lora":
        loraconfig = LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=args.target_modules)
        model.is_gradient_checkpointing = False
        if args.use_fp16:
            model.module = get_peft_model(model.module, loraconfig)
        else:
            model = get_peft_model(model, loraconfig)

    logger.info(f"    Training parts: {args.training_parts}")

    model, opt = initialize(args, logger, model, opt, deepspeed_config)

    diffusion = create_diffusion(
        model=model,
        noise_schedule=args.noise_schedule,
        predict_type=args.predict_type,
        learn_sigma=args.learn_sigma,
        mse_loss_weight_type=args.mse_loss_weight_type,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        noise_offset=args.noise_offset,
    )

    # ===========================================================================
    # Training
    # ===========================================================================

    print(f"    Worker {rank} ready.")

    iters_per_epoch = len(loader)
    logger.info(" ****************************** Running training ******************************")
    logger.info(f"      Number NPUs:               {world_size}")
    logger.info(f"      Number training samples:   {len(dataset):,}")
    logger.info(f"      Number parameters:         {sum(p.numel() for p in model.get_parameters()):,}")
    logger.info(f"      Number trainable params:   {sum(p.numel() for p in get_trainable_params(model)):,}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Iters per epoch:           {iters_per_epoch:,}")
    logger.info(f"      Batch size per device:     {batch_size}")
    logger.info(
        f"      Batch size all device:     {batch_size * world_size * grad_accu_steps:,} (world_size * batch_size * grad_accu_steps)"
    )
    logger.info(f"      Gradient Accu steps:       {args.grad_accu_steps}")
    logger.info(f"      Total optimization steps:  {args.epochs * iters_per_epoch // grad_accu_steps:,}")

    logger.info(f"      Training epochs:           {start_epoch}/{args.epochs}")
    logger.info(f"      Training epoch steps:      {start_epoch_step:,}/{iters_per_epoch:,}")
    logger.info(
        f"      Training total steps:      {train_steps:,}/{min(args.max_training_steps, args.epochs * iters_per_epoch):,}"
    )
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Noise schedule:            {args.noise_schedule}")
    logger.info(f"      Beta limits:               ({args.beta_start}, {args.beta_end})")
    logger.info(f"      Learn sigma:               {args.learn_sigma}")
    logger.info(f"      Prediction type:           {args.predict_type}")
    logger.info(f"      Noise offset:              {args.noise_offset}")

    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Using EMA model:           {args.use_ema} ({args.ema_dtype})")
    if args.use_ema:
        raise NotImplementedError("EMA feature is not supported.")
    logger.info(f"      Using main model fp16:     {args.use_fp16}")
    logger.info(f"      Using extra modules fp16:  {args.extra_fp16}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Experiment directory:      {experiment_dir}")
    logger.info("    *******************************************************************************")

    if args.gc_interval > 0:
        gc.disable()
        gc.collect()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    train_step_for_hunyuan = TrainStepForHunYuan(
        model, opt, diffusion, args.grad_accu_steps, loss_scaler_params, gradient_clipping, iters_per_epoch
    ).set_train(True)

    # Training loop
    epoch = start_epoch
    while epoch < args.epochs:
        # Random shuffle dataset
        shuffle_seed = args.global_seed + epoch
        logger.info(f"    Start random shuffle with seed={shuffle_seed}")
        # Makesure all processors use the same seed to shuffle dataset.
        dataset.shuffle(seed=shuffle_seed, fast=True)
        logger.info("    End of random shuffle")

        # # Move sampler to start_index
        if not args.multireso:
            start_index = start_epoch_step * world_size * batch_size
            if start_index != sampler.start_index:
                sampler.start_index = start_index
                # Reset start_epoch_step to zero, to ensure next epoch will start from the beginning.
                start_epoch_step = 0
                logger.info(f"      Iters left this epoch: {len(loader):,}")

        logger.info(f"    Beginning epoch {epoch}...")
        for batch in loader:
            latents, model_kwargs = prepare_model_inputs(args, batch, vae, text_encoder, text_encoder_t5, freqs_cis_img)

            loss, _ = train_step_for_hunyuan(latents, ms.mutable(model_kwargs))

            if args.use_ema:
                raise NotImplementedError("EMA feature is not supported.")

            # ===========================================================================
            # Log loss values:
            # ===========================================================================
            running_loss += loss.numpy().item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = running_loss / log_steps
                # get lr from deepspeed fused optimizer
                logger.info(
                    f"(step={train_steps:07d}) "
                    + (f"(update_step={train_steps // args.grad_accu_steps:07d}) " if args.grad_accu_steps > 1 else "")
                    + f"Train Loss: {avg_loss:.4f}, "
                    f"Lr: {opt.get_lr().numpy().item():.6g}, "
                    f"Steps/Sec: {steps_per_sec:.2f}, "
                    f"Samples/Sec: {int(steps_per_sec * batch_size * world_size):d}"
                )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # collect gc:
            if args.gc_interval > 0 and (train_steps % args.gc_interval == 0):
                gc.collect()

            if (train_steps % args.ckpt_every == 0 or train_steps % args.ckpt_latest_every == 0) and train_steps > 0:
                save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir, by="step")

            if train_steps >= args.max_training_steps:
                logger.info(f"Breaking step loop at train_steps={train_steps}.")
                break

        if train_steps >= args.max_training_steps:
            logger.info(f"Breaking epoch loop at epoch={epoch}.")
            break

        # Finish an epoch
        if args.ckpt_every_n_epoch > 0 and epoch % args.ckpt_every_n_epoch == 0:
            save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir, by="epoch")

        epoch += 1

    save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir, by="final")


class TrainStepForHunYuan(TrainStep):
    def __init__(
        self,
        model: nn.Cell,
        optimizer: nn.Optimizer,
        diffusion,
        gradient_accumulation_steps,
        loss_scaler_params,
        gradient_clipping,
        length_of_dataloader,
    ):
        super().__init__(
            model,
            optimizer,
            DynamicLossScaler(**loss_scaler_params),
            gradient_clipping,
            gradient_accumulation_steps,
            gradient_accumulation_kwargs=dict(length_of_dataloader=length_of_dataloader),
        )
        self.diffusion = diffusion

    def forward(self, x_start, model_kwargs):
        loss_dict = self.diffusion.training_losses(x_start=x_start, model_kwargs=model_kwargs)
        loss = loss_dict["loss"].mean()
        loss = self.scale_loss(loss)
        return loss, loss_dict  # loss_dict is just a placeholder


if __name__ == "__main__":
    # Start
    main(get_args())
