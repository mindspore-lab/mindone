"""
VideoDiT training pipeline
- Image finetuning
"""
import datetime
import logging
import os
import sys
import time
from typing import Tuple

import yaml
from args_train import parse_args
from data.dataset import create_dataloader

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)


from modules.autoencoder import SD_CONFIG, AutoencoderKL
from modules.dit.video_dit_models import VideoDiT_models
from modules.encoders import FrozenCLIPEmbedder

from examples.dit.pipelines.train_pipeline import DiTWithLoss

# load training modules
# from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindone.utils.seed import set_random_seed

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
) -> Tuple[int, int, int]:
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)

    if distributed:
        device_id = int(os.getenv("DEVICE_ID"))
        ms.set_context(
            mode=mode,
            device_target=device_target,
            device_id=device_id,
            # ascend_config={"precision_mode": "allow_fp32_to_fp16"}, # TODO: tune
        )
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        logger.debug(f"Device_id: {device_id}, rank_id: {rank_id}, device_num: {device_num}")
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        logger.info(dict(zip(var_info, var_value)))

    else:
        device_num = 1
        device_id = int(os.getenv("DEVICE_ID", 0))
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
            device_id=device_id,
            # ascend_config={"precision_mode": "allow_fp32_to_fp16"},  # TODO: tune
        )

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    return device_id, rank_id, device_num


def set_temp_blocks(dit_model, train=True):
    for param in dit_model.get_parameters():  # freeze vae
        if "temp_blocks." in param.name:
            param.requires_grad = train
        else:
            param.requires_grad = False


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.output_path = os.path.join(args.output_path, time_str)

    # 1. init
    device_id, rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. model initiate and weight loading
    # 2.1 dit
    logger.info(f"{args.model_name}-{args.image_size}x{args.image_size} init")
    latent_size = args.image_size // 8
    dit_model = VideoDiT_models[args.model_name](
        input_size=latent_size,
        num_classes=1000,
        dtype=ms.float16 if args.use_fp16 else ms.float32,
        block_kwargs={"enable_flash_attention": args.enable_flash_attention},
        condition=args.condition,
        num_frames=args.num_frames,
    )
    dit_model.load_params_from_dit_ckpt(args.dit_checkpoint)
    # set temp_blocks  train
    set_temp_blocks(dit_model, train=True)

    # 2.2 vae
    logger.info("vae init")
    vae = AutoencoderKL(
        SD_CONFIG,
        4,
        ckpt_path=args.vae_checkpoint,
        use_fp16=False,  # disable amp for vae
    )
    vae = vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False
    if args.condition == "text":
        text_encoder = FrozenCLIPEmbedder(
            use_fp16=True,
            tokenizer_name="BpeTokenizer",
            context_length=77,
            vocab_size=49408,
            output_dim=768,
            width=768,
            layers=12,
            heads=12,
            epsilon=1e-5,
            use_quick_gelu=True,
        )
    else:
        text_encoder = None
    dit_model_with_loss = DiTWithLoss(
        dit_model, vae, text_encoder=text_encoder, scale_factor=args.sd_scale_factor, condition=args.condition
    )
    # video dataset
    data_config = dict(
        video_folder=args.data_path,
        csv_path=args.data_path + "/video_caption.csv",
        sample_size=args.image_size,
        sample_stride=args.frame_stride,
        sample_n_frames=args.num_frames,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_parallel_workers=args.num_parallel_workers,
        max_rowsize=64,
    )

    dataset = create_dataloader(
        data_config,
        tokenizer=None,
        is_image=False,
        device_num=device_num,
        rank_id=rank_id,
        condition_column=args.condition,
    )
    dataset_size = dataset.get_dataset_size()

    # 4. build training utils: lr, optim, callbacks, trainer
    # build learning rate scheduler
    if not args.decay_steps:
        args.decay_steps = args.epochs * dataset_size - args.warmup_steps  # fix lr scheduling
        if args.decay_steps <= 0:
            logger.warning(
                f"decay_steps is {args.decay_steps}, please check epochs, dataset_size and warmup_steps. "
                f"Will force decay_steps to be set to 1."
            )
            args.decay_steps = 1

    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        name=args.scheduler,
        lr=args.start_learning_rate,
        end_lr=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        num_epochs=args.epochs,
    )

    # build optimizer
    optimizer = create_optimizer(
        dit_model_with_loss.trainable_params(),
        name=args.optim,
        betas=args.betas,
        eps=args.optim_eps,
        group_strategy=args.group_strategy,
        weight_decay=args.weight_decay,
        lr=lr,
    )

    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        raise ValueError

    # resume ckpt
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    start_epoch = 0
    if args.resume:
        resume_ckpt = os.path.join(ckpt_dir, "train_resume.ckpt") if isinstance(args.resume, bool) else args.resume

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(dit_model, optimizer, resume_ckpt)
        loss_scaler.loss_scale_value = loss_scale
        loss_scaler.cur_iter = cur_iter
        loss_scaler.last_overflow_iter = last_overflow_iter

    # trainer (standalone and distributed)
    ema = (
        EMA(
            dit_model,
            ema_decay=0.9999,
        )
        if args.use_ema
        else None
    )

    net_with_grads = TrainOneStepWrapper(
        dit_model_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    # model = Model(net_with_grads)
    # callbacks
    # callback = [TimeMonitor(args.callback_size)]
    # ofm_cb = OverflowMonitor()
    # callback.append(ofm_cb)

    # if rank_id == 0:
    #     save_cb = EvalSaveCallback(
    #         network=dit_model,
    #         rank_id=rank_id,
    #         ckpt_save_dir=ckpt_dir,
    #         ema=ema,
    #         ckpt_save_policy="latest_k",
    #         ckpt_max_keep=args.ckpt_max_keep,
    #         step_mode=args.step_mode,
    #         ckpt_save_interval=args.ckpt_save_interval,
    #         log_interval=args.callback_size,
    #         start_epoch=start_epoch,
    #         model_name="sd" if args.image_finetune else "ad",
    #         use_lora=args.motion_lora_finetune,
    #         lora_rank=args.motion_lora_rank,
    #         param_save_filter=[".temporal_transformer."] if args.save_mm_only else None,
    #         record_lr=False,  # TODO: check LR retrival for new MS on 910b
    #     )
    #     callback.append(save_cb)
    #     if args.profile:
    #         callback.append(ProfilerCallback())

    # 5. log and save config
    if rank_id == 0:
        # 4. print key info
        num_params_vae, num_params_vae_trainable = count_params(vae)
        num_params_dit, num_params_dit_trainable = count_params(dit_model)
        num_params = num_params_vae + num_params_dit
        num_params_trainable = num_params_vae_trainable + num_params_dit_trainable
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"Data path: {args.data_path}",
                f"Num params: {num_params:,} (dit: {num_params_dit:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"Precision: {dit_model.dtype}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.train_batch_size}",
                f"Image size: {args.image_size}",
                f"Frames: {args.num_frames}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"Enable flash attention: {args.enable_flash_attention}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    # 6. train
    # model.train(
    #     args.epochs,
    #     dataset,
    #     callbacks=callback,
    #     dataset_sink_mode=args.dataset_sink_mode,
    #     sink_size=args.sink_size,
    #     initial_epoch=start_epoch,
    # )
    if args.dataset_sink_mode:
        raise ValueError("dataset sink = True not supported now!")
    # use training for loop
    train_class2video(dit_model, args, net_with_grads, dataset, rank_id, optimizer=optimizer)


def train_class2video(
    model,
    args,
    train_step_fn,
    dataloader,
    rank_id,
    start_epoch=0,
    optimizer=None,
):
    total_step = len(dataloader) * args.epochs
    # 3. training loop
    if args.mode == 0:
        logger.info(
            "The first step will compile the graph, which may take longer time; " "You can come back later :)",
        )
    for i_epoch in range(start_epoch, args.epochs):
        # 3.1 train one epoch
        train_one_epoch(
            model,
            i_epoch,
            args,
            train_step_fn,
            dataloader,
            optimizer,
            total_step,
            rank_id,
        )


def train_one_epoch(
    model,
    i_epoch,
    args,
    train_step_fn,
    dataloader,
    optimizer,
    total_step,
    rank_id,
):
    s_time = time.time()
    for i, data in enumerate(dataloader):
        set_temp_blocks(model, train=True)
        i_step = i + i_epoch * len(dataloader) + 1
        image, cond = data
        if args.condition == "text":
            model_args = [cond, None]
        elif args.condition == "class":
            model_args = [None, cond]
        # Train a step
        loss, overflow, _ = train_step_fn(image, *model_args)
        if overflow:
            if not args.drop_overflow_update:
                logger.info(f"Step {i_step}/{total_step}, overflow, still update.")
            else:
                logger.info(f"Step {i_step}/{total_step}, overflow, skip.")

        set_temp_blocks(model, train=False)
        # Print meg
        if i_step % args.log_interval == 0 and rank_id % 8 == 0:
            if optimizer.dynamic_lr:
                cur_lr = optimizer.learning_rate(Tensor(i_step - 1, ms.int32)).asnumpy().item()
            else:
                cur_lr = optimizer.learning_rate.asnumpy().item()
            logger.info(
                f"Step {i_step}/{total_step}, lr: {cur_lr}, loss: {loss.asnumpy()[0]:.6f}"
                f", time cost: {(time.time()-s_time) * 1000 / args.log_interval:.2f} ms",
            )
            s_time = time.time()

        # Save checkpoint
        if i_step % args.ckpt_save_interval == 0 and rank_id % 8 == 0:
            save_ckpt_dir = os.path.join(args.output_path, "ckpt")
            if not os.path.exists(save_ckpt_dir):
                os.makedirs(save_ckpt_dir)
            save_filename = f"VideoDiT-{i_step}.ckpt"
            ms.save_checkpoint(model, os.path.join(save_ckpt_dir, save_filename))


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
