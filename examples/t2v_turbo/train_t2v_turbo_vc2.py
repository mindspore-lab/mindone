import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import nn, ops
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from data.dataset import create_dataloader
from ode_solver import DDIMSolver
from pipeline.lcd_with_loss import LCDWithLoss
from reward_fn import get_reward_fn
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from utils.checkpoint import CheckpointManager
from utils.common_utils import load_model_checkpoint
from utils.env import init_env
from utils.lora_handler import LoraHandler
from utils.utils import freeze_params, instantiate_from_config

from examples.t2v_turbo.configs.train_args import parse_args
from mindone.diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from mindone.diffusers.training_utils import set_seed
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.recorder import PerfRecorder
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.logger import set_logger

sys.path.append("./mindone/examples/stable_diffusion_xl")
from gm.modules.embedders.open_clip.tokenizer import tokenize

logger = logging.getLogger(__name__)


def _to_abspath(rp):
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(__dir__, rp)


def main(args):
    args = parse_args()
    ms.set_context(mode=args.mode, jit_syntax_level=ms.STRICT)
    dtype_map = {"no": ms.float32, "fp32": ms.float32, "fp16": ms.float16, "bf16": ms.bfloat16}
    dtype = dtype_map[args.mixed_precision]
    rank_id, device_num = init_env(
        args.mode,
        args.seed,
        args.use_parallel,
        parallel_mode="data",
        device_target=args.device_target,
        jit_level=args.jit_level,
        global_bf16=args.global_bf16,
        debug=args.debug,
        dtype=dtype,
    )

    logging_dir = Path(args.output_dir, args.logging_dir)
    set_logger(name="", output_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load teacher Model
    config = OmegaConf.load(args.pretrained_model_cfg)
    model_config = config.pop("model", OmegaConf.create())
    model_config["params"]["cond_stage_config"]["params"]["pretrained_ckpt_path"] = args.pretrained_enc_path
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = pretrained_t2v.to_float(dtype)
    pretrained_t2v = load_model_checkpoint(
        pretrained_t2v,
        args.pretrained_model_path,
    )

    vae = pretrained_t2v.first_stage_model
    vae_scale_factor = model_config["params"]["scale_factor"]
    text_encoder = pretrained_t2v.cond_stage_model
    teacher_unet = pretrained_t2v.model.diffusion_model

    # Freeze teacher vae, text_encoder, and teacher_unet
    freeze_params(vae)
    freeze_params(text_encoder)
    freeze_params(teacher_unet)

    # Create online student U-Net. This will be updated by the optimizer (e.g. via backpropagation.)
    # Add `time_cond_proj_dim` to the student U-Net if `teacher_unet.config.time_cond_proj_dim` is None
    time_cond_proj_dim = (
        teacher_unet.time_cond_proj_dim if teacher_unet.time_cond_proj_dim is not None else args.unet_time_cond_proj_dim
    )
    unet_config = model_config["params"]["unet_config"]
    unet_config["params"]["time_cond_proj_dim"] = time_cond_proj_dim
    unet_config["params"]["use_checkpoint"] = args.use_recompute
    unet = instantiate_from_config(unet_config)
    # load teacher_unet weights into unet
    ms.load_param_into_net(unet, teacher_unet.parameters_dict(), strict_load=False)
    freeze_params(unet)
    unet.set_train(True)

    use_unet_lora = True
    lora_manager = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=use_unet_lora,
        save_for_webui=True,
        unet_replace_modules=["UNetModel"],
    )

    unet_lora_params, unet_negation = lora_manager.add_lora_to_model(
        use_unet_lora,
        unet,
        lora_manager.unet_replace_modules,
        lora_path=args.pretrained_lora_path,
        dropout=args.lora_dropout,
        r=args.lora_rank,
    )

    if args.reward_scale > 0:
        reward_fn = get_reward_fn(
            args.reward_fn_name,
            precision=args.mixed_precision,
            rm_ckpt_dir=args.image_rm_ckpt_dir,
        )
    else:
        reward_fn = None

    if args.video_reward_scale > 0:
        video_rm_fn = get_reward_fn(
            args.video_rm_name,
            precision=args.mixed_precision,
            rm_ckpt_dir=args.video_rm_ckpt_dir,
            n_frames=args.video_rm_batch_size,
        )
    else:
        video_rm_fn = None

    # Create the noise scheduler and the desired noise schedule.
    noise_scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )

    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    alpha_schedule = ops.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = ops.sqrt(1 - noise_scheduler.alphas_cumprod)
    # Initialize the DDIM ODE solver for distillation.

    if args.no_scale_pred_x0:
        use_scale = False
    else:
        use_scale = model_config["params"]["use_scale"]

    assert not use_scale
    scale_b = model_config["params"]["scale_b"]
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        ddim_timesteps=args.num_ddim_timesteps,
        use_scale=use_scale,
        scale_b=scale_b,
        ddim_eta=args.ddim_eta,
    )

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unet.dtype != ms.float32:
        raise ValueError(f"Controlnet loaded as datatype {unet.dtype}. {low_precision_error_string}")

    # Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = ms.float32
    if args.mixed_precision == "fp16":
        weight_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = ms.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae = vae.to_float(weight_dtype)
    text_encoder = text_encoder.to_float(weight_dtype)

    # Move teacher_unet to device, optionally cast to weight_dtype
    if args.cast_teacher_unet:
        teacher_unet = teacher_unet.to_float(weight_dtype)

    # Make sure the trainable params are in float32.
    models = [unet]

    # Print trainable parameters statistics
    for peft_model in models:
        all_params = sum(p.numel() for p in peft_model.get_parameters())
        trainable_params = sum(p.numel() for p in peft_model.trainable_params())
        logger.info(
            f"{peft_model.__class__.__name__:<30s} ==> Trainable params: {trainable_params:<10,d} || "
            f"All params: {all_params:<16,d} || Trainable ratio: {trainable_params / all_params:.8%}"
        )

    # Dataset creation and data processing
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.

    num_workers = args.dataloader_num_workers
    resolution = tuple([s * 8 for s in model_config["params"]["image_size"]])
    csv_path = args.csv_path if args.csv_path is not None else os.path.join(args.data_path, "video_caption.csv")
    data_config = dict(
        video_folder=_to_abspath(args.data_path),
        csv_path=_to_abspath(csv_path),
        sample_size=resolution,
        sample_stride=1,  # args.frame_stride,
        sample_n_frames=args.n_frames,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_parallel_workers=args.dataloader_num_workers,
        max_rowsize=64,
        random_drop_text=False,
    )

    train_dataloader = create_dataloader(
        data_config,
        tokenizer=tokenize,
        is_image=False,
        device_num=device_num,
        rank_id=rank_id,
        n_samples=args.max_train_samples,
    )

    num_train_examples = args.max_train_samples
    global_batch_size = args.train_batch_size * device_num
    num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))

    train_dataloader.num_batches = num_worker_batches * args.dataloader_num_workers
    train_dataloader.num_samples = train_dataloader.num_batches * global_batch_size

    # 14. LR Scheduler creation
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = create_scheduler(
        steps_per_epoch=num_update_steps_per_epoch,
        name=args.scheduler,
        lr=args.learning_rate,
        end_lr=args.end_learning_rate,
        warmup_steps=args.lr_warmup_steps,
        decay_steps=args.decay_steps,
        num_epochs=args.num_train_epochs,
    )

    # Prepare for training
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    uncond_prompt, _ = tokenize([""] * args.train_batch_size)
    uncond_prompt = ms.Tensor(np.array(uncond_prompt, dtype=np.int32))
    uncond_prompt_embeds = text_encoder(uncond_prompt)
    if isinstance(uncond_prompt_embeds, DiagonalGaussianDistribution):
        uncond_prompt_embeds = uncond_prompt_embeds.mode()

    lcd_with_loss = LCDWithLoss(
        vae=vae,
        text_encoder=text_encoder,
        teacher_unet=teacher_unet,
        unet=unet,
        noise_scheduler=noise_scheduler,
        alpha_schedule=alpha_schedule,
        sigma_schedule=sigma_schedule,
        weight_dtype=weight_dtype,
        vae_scale_factor=vae_scale_factor,
        time_cond_proj_dim=time_cond_proj_dim,
        args=args,
        solver=solver,
        uncond_prompt_embeds=uncond_prompt_embeds,
        reward_fn=reward_fn,
        video_rm_fn=video_rm_fn,
    ).set_train()

    # Optimizer creation
    optimizer = create_optimizer(
        lcd_with_loss.unet.trainable_params(),
        name="adamw",
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        group_strategy="norm_and_bias",
        weight_decay=args.adam_weight_decay,
        lr=lr_scheduler,
    )

    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale,
            scale_factor=args.loss_scale_factor,
            scale_window=args.scale_window,
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        raise ValueError

    # Trainer
    ema = EMA(lcd_with_loss.unet, ema_decay=args.ema_decay, offloading=True) if args.use_ema else None

    net_with_grads = TrainOneStepWrapper(
        lcd_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    start_epoch = 0

    # 16. Train!
    total_batch_size = args.train_batch_size * device_num * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {train_dataloader.num_batches}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    if rank_id == 0:
        ckpt_folder = args.output_dir + "/ckpt"
        ckpt_manager = CheckpointManager(
            ckpt_folder, "latest_k", k=args.checkpoints_total_limit, lora_manager=lora_manager
        )
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            os.makedirs(ckpt_folder)

        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)

        perf_columns = ["step", "loss", "train_time(s)", "shape"]
        if start_epoch == 0:
            record = PerfRecorder(args.output_dir, metric_names=perf_columns)
        else:
            record = PerfRecorder(args.output_dir, resume=True)

    global_step = 0
    ds_iter = train_dataloader.create_tuple_iterator(num_epochs=args.num_train_epochs - start_epoch)

    for epoch in range(start_epoch + 1, args.num_train_epochs + 1):
        for step, data in enumerate(ds_iter, 1):
            start_time_s = time.time()
            loss, overflow, scaling_sens = net_with_grads(*data)
            step_time = time.time() - start_time_s

            global_step += 1

            if step % args.log_interval == 0:
                loss = float(loss.asnumpy())
                logger.info(
                    f"Epoch: {epoch}, Step: {step}, Global step {global_step}, Loss: {loss:.5f}, Step time: {step_time*1000:.2f}ms"
                )

            if overflow:
                logger.warning("overflow detected")

            if rank_id == 0:
                step_perf_value = [global_step, loss, step_time]
                record.add(*step_perf_value)

            start_time_s = time.time()

        if (epoch % args.ckpt_save_interval == 0) or (epoch == args.num_train_epochs):
            ckpt_name = f"t2v-turbo-e{epoch}.ckpt"
            if ema is not None:
                ema.swap_before_eval()

            ckpt_manager.save(lcd_with_loss.unet, None, ckpt_name=ckpt_name)
            if ema is not None:
                ema.swap_after_eval()


if __name__ == "__main__":
    args = parse_args()
    main(args)
