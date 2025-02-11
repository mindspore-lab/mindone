"""
AnimateDiff training pipeline
- Image finetuning
- Motion module training
"""
import datetime
import logging
import math
import os
import shutil
import sys
from typing import Tuple

import yaml
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import Model, load_checkpoint, load_param_into_net, nn
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from ad.data.dataset import create_dataloader

# from ad.data.dataset import check_sanity
from ad.utils.load_models import load_motion_modules, update_unet2d_params_for_unet3d
from args_train import parse_args

from mindone.models.lora import inject_trainable_lora, make_only_lora_params_trainable
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper

# from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import get_obj_from_str
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params, load_param_into_net_with_filter
from mindone.utils.seed import set_random_seed
from mindone.utils.version_control import is_old_ms_version

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"
os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"

logger = logging.getLogger(__name__)


def _to_abspath(rp):
    return os.path.join(__dir__, rp)


def build_model_from_config(config, unet_config_update=None, vae_use_fp16=None, snr_gamma=None):
    config = OmegaConf.load(config).model
    if unet_config_update is not None:
        # config["params"]["unet_config"]["params"]["enable_flash_attention"] = enable_flash_attention
        unet_args = config["params"]["unet_config"]["params"]
        for name, value in unet_config_update.items():
            if value is not None:
                logger.info("Arg `{}` updated: {} -> {}".format(name, unet_args[name], value))
                unet_args[name] = value

    if vae_use_fp16 is not None:
        config.params.first_stage_config.params.use_fp16 = vae_use_fp16

    if snr_gamma is not None:
        config.params.snr_gamma = snr_gamma

    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    config_params = config.get("params", dict())
    # config_params['cond_stage_trainable'] = cond_stage_trainable # TODO: easy config
    return get_obj_from_str(config["target"])(**config_params)


def load_pretrained_model(
    pretrained_ckpt, net, unet_initialize_random=False, load_unet3d_from_2d=False, unet3d_type="adv2"
):
    logger.info(f"Loading pretrained model from {pretrained_ckpt}")
    if os.path.exists(pretrained_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)

        if load_unet3d_from_2d:
            param_dict = update_unet2d_params_for_unet3d(param_dict, unet3d_type=unet3d_type)

        if unet_initialize_random:
            pnames = list(param_dict.keys())
            # pop unet params from pretrained weight
            for pname in pnames:
                if pname.startswith("model.diffusion_model"):
                    param_dict.pop(pname)
            logger.warning("UNet will be initialized randomly")

        if is_old_ms_version():
            param_not_load = load_param_into_net(net, param_dict, filter=param_dict.keys())
        else:
            param_not_load, ckpt_not_load = load_param_into_net_with_filter(net, param_dict, filter=param_dict.keys())

        logger.info(
            "Net params not load: {}, Total net params not loaded: {}".format(param_not_load, len(param_not_load))
        )
        logger.info(
            "Ckpt params not load: {}, Total ckpt params not loaded: {}".format(ckpt_not_load, len(ckpt_not_load))
        )

        if not unet_initialize_random:
            assert (
                len(ckpt_not_load) == 0
            ), "All params in ckpt should be loaded to the network. See log for detailed missing params."
    else:
        logger.warning(f"Checkpoint file {pretrained_ckpt} dose not exist!!!")


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

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if distributed:
        device_id = int(os.getenv("DEVICE_ID"))
        ms.set_context(
            mode=mode,
            device_target=device_target,
            device_id=device_id,
            ascend_config={"precision_mode": "allow_fp32_to_fp16"},  # TODO: tune
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
            ascend_config={"precision_mode": "allow_fp32_to_fp16"},  # TODO: tune
        )

    return device_id, rank_id, device_num


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

    # 2. build model
    unet_config_update = dict(
        enable_flash_attention=args.enable_flash_attention,
        use_recompute=args.use_recompute,
        recompute_strategy=args.recompute_strategy,
    )
    latent_diffusion_with_loss = build_model_from_config(
        _to_abspath(args.model_config),
        unet_config_update,
        vae_use_fp16=args.vae_fp16,
        snr_gamma=args.snr_gamma,
    )
    # 1) load sd pretrained weight
    load_pretrained_model(
        _to_abspath(args.pretrained_model_path),
        latent_diffusion_with_loss,
        unet_initialize_random=args.unet_initialize_random,
        load_unet3d_from_2d=(not args.image_finetune),
        unet3d_type="adv1" if "mmv1" in args.model_config else "adv2",  # TODO: better not use filename to judge version
    )

    # TODO: debugging
    # latent_diffusion_with_loss = auto_mixed_precision(latent_diffusion_with_loss, "O2")

    if not args.image_finetune:
        # load mm pretrained weight
        if args.motion_module_path != "":
            load_motion_modules(latent_diffusion_with_loss, _to_abspath(args.motion_module_path))

        # set motion module amp O2 if required for memory reduction
        if args.force_motion_module_amp_O2:
            logger.warning("Force to set motion module in amp level O2")
            latent_diffusion_with_loss.model.diffusion_model.set_mm_amp_level("O2")

        # inject lora dense layers to motion modules if set
        if args.motion_lora_finetune:
            # for param in latent_diffusion_with_loss.get_parameters():
            #     param.requires_grad = False
            motion_lora_layers, _ = inject_trainable_lora(
                latent_diffusion_with_loss,
                rank=args.motion_lora_rank,
                use_fp16=True,
                scale=args.motion_lora_alpha,
                target_modules=["ad.modules.diffusionmodules.motion_module.VersatileAttention"],
            )
            trainable_params = make_only_lora_params_trainable(latent_diffusion_with_loss)
            logging.info(
                "Motion lora layers injected. Num lora layers: {}, Num lora params: {}".format(
                    len(motion_lora_layers), len(trainable_params)
                )
            )
        else:
            # set only motion module trainable for mm finetuning
            num_mm_trainable = 0
            for param in latent_diffusion_with_loss.model.get_parameters():
                # exclude positional embedding params from training
                if (".temporal_transformer." in param.name) and (".pe" not in param.name):
                    param.requires_grad = True
                    num_mm_trainable += 1
                else:
                    param.requires_grad = False
            logger.info("Num MM trainable params {}".format(num_mm_trainable))
            # assert num_mm_trainable in [546, 520], "Expect 546 trainable params for MM-v2 or 520 for MM-v1."

    # count total params and trainable params
    tot_params, trainable_params = count_params(latent_diffusion_with_loss.model)
    logger.info("UNet3D: total param size {:,}, trainable {:,}".format(tot_params, trainable_params))
    assert trainable_params > 0, "No trainable parameters. Please check model config."

    # 3. build dataset
    csv_path = args.csv_path if args.csv_path is not None else os.path.join(args.data_path, "video_caption.csv")
    if args.image_finetune:
        logger.info("Task is image finetune, num_frames and frame_stride is forced to 1")
        args.num_frames = 1
        args.frame_stride = 1
        data_config = dict(
            video_folder=_to_abspath(args.data_path),
            csv_path=_to_abspath(csv_path),
            sample_size=args.image_size,
            sample_stride=args.frame_stride,
            sample_n_frames=args.num_frames,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_parallel_workers=args.num_parallel_workers,
            max_rowsize=32,
            random_drop_text=args.random_drop_text,
            random_drop_text_ratio=args.random_drop_text_ratio,
            video_column=args.video_column,
            caption_column=args.caption_column,
            train_data_type=args.train_data_type,
            disable_flip=args.disable_flip,
        )
    else:
        data_config = dict(
            video_folder=_to_abspath(args.data_path),
            csv_path=_to_abspath(csv_path),
            sample_size=args.image_size,
            sample_stride=args.frame_stride,
            sample_n_frames=args.num_frames,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_parallel_workers=args.num_parallel_workers,
            max_rowsize=64,
            random_drop_text=args.random_drop_text,
            random_drop_text_ratio=args.random_drop_text_ratio,
            video_column=args.video_column,
            caption_column=args.caption_column,
            train_data_type=args.train_data_type,
            disable_flip=args.disable_flip,
        )

    tokenizer = latent_diffusion_with_loss.cond_stage_model.tokenize
    dataset = create_dataloader(
        data_config, tokenizer=tokenizer, is_image=args.image_finetune, device_num=device_num, rank_id=rank_id
    )
    dataset_size = dataset.get_dataset_size()

    # compute total steps and data epochs (in unit of data sink size)
    if args.train_steps == -1:
        assert args.epochs != -1
        total_train_steps = args.epochs * dataset_size
    else:
        total_train_steps = args.train_steps

    if args.dataset_sink_mode and args.sink_size != -1:
        steps_per_sink = args.sink_size
    else:
        steps_per_sink = dataset_size
    sink_epochs = math.ceil(total_train_steps / steps_per_sink)

    if args.ckpt_save_steps == -1:
        ckpt_save_interval = args.ckpt_save_epochs
        step_mode = False
    else:
        step_mode = not args.dataset_sink_mode
        if not args.dataset_sink_mode:
            ckpt_save_interval = args.ckpt_save_steps
        else:
            # still need to count interval in sink epochs
            ckpt_save_interval = max(1, args.ckpt_save_steps // steps_per_sink)
            if args.ckpt_save_steps % steps_per_sink != 0:
                logger.warning(
                    f"`ckpt_save_steps` must be times of sink size or dataset_size under dataset sink mode."
                    f"Checkpoint will be saved every {ckpt_save_interval * steps_per_sink} steps."
                )
    step_mode = step_mode if args.step_mode is None else args.step_mode

    logger.info(f"train_steps: {total_train_steps}, train_epochs: {args.epochs}, sink_size: {args.sink_size}")
    logger.info(f"total train steps: {total_train_steps}, sink epochs: {sink_epochs}")
    logger.info(
        "ckpt_save_interval: {} {}".format(
            ckpt_save_interval, "steps" if (not args.dataset_sink_mode and step_mode) else "sink epochs"
        )
    )

    # if args.dataset_sink_mode:
    #    if os.environ.get("MS_DATASET_SINK_QUEUE") is None:
    #        os.environ["MS_DATASET_SINK_QUEUE"] = "10"
    #        print("WARNING: Set env `MS_DATASET_SINK_QUEUE` to 10.")
    #    else:
    #        print("D--: get dataset sink queue: ", os.environ.get("MS_DATASET_SINK_QUEUE") )

    # 4. build training utils: lr, optim, callbacks, trainer
    # build learning rate scheduler
    if not args.decay_steps:
        args.decay_steps = total_train_steps - args.warmup_steps  # fix lr scheduling
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
        total_steps=total_train_steps,
    )

    # build optimizer
    optimizer = create_optimizer(
        latent_diffusion_with_loss.trainable_params(),
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

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
            latent_diffusion_with_loss, optimizer, resume_ckpt
        )
        loss_scaler.loss_scale_value = loss_scale
        loss_scaler.cur_iter = cur_iter
        loss_scaler.last_overflow_iter = last_overflow_iter
        logger.info(f"Resume training from {resume_ckpt}")

    # trainer (standalone and distributed)
    ema = (
        EMA(
            latent_diffusion_with_loss,
            ema_decay=0.9999,
        )
        if args.use_ema
        else None
    )

    net_with_grads = TrainOneStepWrapper(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    model = Model(net_with_grads)

    # callbacks
    callback = [TimeMonitor(args.log_interval)]
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)

    if rank_id == 0:
        save_cb = EvalSaveCallback(
            network=latent_diffusion_with_loss,
            rank_id=rank_id,
            ckpt_save_dir=ckpt_dir,
            ema=ema,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=args.ckpt_max_keep,
            step_mode=step_mode,
            use_step_unit=(args.ckpt_save_steps != -1),
            ckpt_save_interval=ckpt_save_interval,
            log_interval=args.log_interval,
            start_epoch=start_epoch,
            model_name="sd" if args.image_finetune else "ad",
            use_lora=args.motion_lora_finetune,
            lora_rank=args.motion_lora_rank,
            param_save_filter=[".temporal_transformer."] if args.save_mm_only else None,
            record_lr=False,  # TODO: check LR retrival for new MS on 910b
        )
        callback.append(save_cb)
        if args.profile:
            callback.append(ProfilerCallback())

    # 5. log and save config
    if rank_id == 0:
        num_params_unet, _ = count_params(latent_diffusion_with_loss.model.diffusion_model)
        num_params_text_encoder, _ = count_params(latent_diffusion_with_loss.cond_stage_model)
        num_params_vae, _ = count_params(latent_diffusion_with_loss.first_stage_model)
        num_params, num_trainable_params = count_params(latent_diffusion_with_loss)

        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"Data path: {args.data_path}",
                f"Num params: {num_params:,} (unet: {num_params_unet:,}, text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_trainable_params:,}",
                f"Precision: {latent_diffusion_with_loss.model.diffusion_model.dtype}",
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
                f"Random drop text: {args.random_drop_text}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")
        # backup config files
        shutil.copyfile(args.model_config, os.path.join(args.output_path, os.path.basename(args.model_config)))

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    # 6. train
    # TODO: start_epoch already recorded in sink size?
    model.train(
        sink_epochs,
        dataset,
        callbacks=callback,
        dataset_sink_mode=args.dataset_sink_mode,
        sink_size=args.sink_size,
        initial_epoch=start_epoch,
    )


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
