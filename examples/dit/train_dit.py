"""
DiT training pipeline
- Image finetuning conditioned on class labels (optional)
"""
import argparse
import datetime
import logging
import os
import sys

import yaml
from data.imagenet_dataset import create_dataloader_imagenet
from pipelines.train_pipeline import DiTWithLoss
from utils.model_utils import load_dit_ckpt_params, str2bool

import mindspore as ms
from mindspore import Model, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)


from diffusion import create_diffusion
from mindcv.optim.adamw import AdamW
from modules.autoencoder import SD_CONFIG, AutoencoderKL

from mindone.models.dit import DiT_models

# load training modules
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor
from mindone.trainers.ema import EMA
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.env import init_train_env
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)


def set_dit_all_params(dit_model, train=True, **kwargs):
    n_params_trainable = 0
    for param in dit_model.get_parameters():
        param.requires_grad = train
        if train:
            n_params_trainable += 1
    logger.info(f"Set {n_params_trainable} params to train.")


def set_dit_params(dit_model, ft_all_params, **kwargs):
    if ft_all_params:
        set_dit_all_params(dit_model, **kwargs)
    else:
        raise ValueError("Fintuning partial params is not supported!")


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.output_path = os.path.join(args.output_path, time_str)

    # 1. init mindspore env
    _, rank_id, device_num = init_train_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
    )
    if args.mode == ms.GRAPH_MODE:
        jit_level = "O1"
        ms.set_context(jit_config={"jit_level": jit_level})
        logger.info(f"set jit_level: {jit_level}.")

    # set logger path
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. model initiate and weight loading
    # 2.1 dit
    logger.info(f"{args.model_name}-{args.image_size}x{args.image_size} init")
    latent_size = args.image_size // 8
    dit_model = DiT_models[args.model_name](
        input_size=latent_size,
        num_classes=args.num_classes,
        block_kwargs={"enable_flash_attention": args.enable_flash_attention},
        use_recompute=args.use_recompute,
    )
    if args.dtype == "fp16":
        model_dtype = ms.float16
        dit_model = auto_mixed_precision(dit_model, amp_level="O2", dtype=model_dtype)
    elif args.dtype == "bf16":
        model_dtype = ms.bfloat16
        dit_model = auto_mixed_precision(dit_model, amp_level="O2", dtype=model_dtype)
    else:
        model_dtype = ms.float32

    if args.dit_checkpoint:
        dit_model = load_dit_ckpt_params(dit_model, args.dit_checkpoint)
    else:
        logger.info("Initialize DIT ramdonly")
    # set dit params to train
    dit_model.set_train(True)
    set_dit_params(dit_model, ft_all_params=True, train=True)

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

    # 2.3 diffusion and NetworkWithLoss
    diffusion = create_diffusion(timestep_respacing="")
    latent_diffusion_with_loss = DiTWithLoss(
        dit_model,
        vae,
        diffusion,
        args.sd_scale_factor,
        args.condition,
        text_encoder=None,
        cond_stage_trainable=False,
    )

    # image dataset
    data_config = dict(
        data_folder=args.data_path,
        sample_size=args.image_size,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_parallel_workers=args.num_parallel_workers,
    )
    dataset = create_dataloader_imagenet(
        data_config,
        device_num=device_num,
        rank_id=rank_id,
    )

    dataset_size = dataset.get_dataset_size()

    # 4. build training utils: lr, optim, callbacks, trainer
    # build optimizer
    optimizer = AdamW(
        latent_diffusion_with_loss.trainable_params(), learning_rate=1e-4, beta1=0.9, beta2=0.999, weight_decay=0.0
    )
    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        raise ValueError
    # trainer (standalone and distributed)
    ema = EMA(
        latent_diffusion_with_loss.network,
        ema_decay=0.9999,
    )
    net_with_grads = TrainOneStepWrapper(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ema=ema,
    )

    model = Model(net_with_grads)
    # callbacks
    callback = [TimeMonitor(args.log_interval)]
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    start_epoch = 0

    if rank_id == 0:
        if not args.dataset_sink_mode:
            log_interval = args.log_interval
        else:
            log_interval = dataset_size if args.sink_size == -1 else args.sink_size
        save_cb = EvalSaveCallback(
            network=latent_diffusion_with_loss.network,  # save dit only
            rank_id=rank_id,
            ckpt_save_dir=ckpt_dir,
            ema=ema,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=args.ckpt_max_keep,
            step_mode=args.step_mode,
            ckpt_save_interval=args.ckpt_save_interval,
            log_interval=log_interval,
            start_epoch=start_epoch,
            model_name="DiT",
            record_lr=True,
        )
        callback.append(save_cb)

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
                f"Use model dtype: {model_dtype}",
                f"Batch size: {args.train_batch_size}",
                f"Image size: {args.image_size}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Total training steps: {dataset_size * args.epochs:,}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Enable flash attention: {args.enable_flash_attention}",
                f"Dataset sink: {args.dataset_sink_mode}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    # 6. train
    model.train(
        args.epochs,
        dataset,
        callbacks=callback,
        dataset_sink_mode=args.dataset_sink_mode,
        sink_size=args.sink_size,
        initial_epoch=start_epoch,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    # the following args's defualt value will be overrided if specified in config yaml
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")

    # ms
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--seed", default=3407, type=int, help="data path")

    # training hyper-params
    parser.add_argument("--train_batch_size", default=10, type=int, help="batch size")
    parser.add_argument("--dataset_sink_mode", default=False, type=str2bool, help="sink mode")
    parser.add_argument("--sink_size", default=-1, type=int, help="dataset sink size. If -1, sink size = dataset size.")
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="epochs. If dataset_sink_mode is on, epochs is with respect to dataset sink size. Otherwise, it's w.r.t the dataset size.",
    )
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000, type=float, help="scale window")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")

    parser.add_argument(
        "--use_recompute",
        default=None,
        type=str2bool,
        help="whether use recompute.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
    )

    parser.add_argument(
        "--num_classes",
        default=1000,
        type=int,
        help="The number of classes of DiT",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="DiT-XL/2",
        help="Model name , such as DiT-XL/2, DiT-L/2",
    )
    parser.add_argument(
        "--dit_checkpoint", type=str, default="models/DiT-XL-2-256x256.ckpt", help="the path to the DiT checkpoint."
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-mse.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )

    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )
    parser.add_argument(
        "--enable_flash_attention",
        default=None,
        type=str2bool,
        help="whether to enable flash attention.",
    )
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--loss_scaler_type", default="dynamic", type=str, help="dynamic or static")

    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs or steps")
    parser.add_argument("--ckpt_max_keep", default=10, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument(
        "--step_mode",
        default=False,
        type=str2bool,
        help="whether save ckpt by steps. If False, save ckpt by epochs.",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument("--image_size", default=256, type=int, help="image size")
    parser.add_argument(
        "--condition",
        default=None,
        type=str,
        help="the condition types: `None` means using no conditions; `text` means using text embedding as conditions;"
        " `class` means using class labels as conditions."
        "DiT only supports `class`condition",
    )
    parser.add_argument("--num_parallel_workers", default=12, type=int, help="num workers for data loading")
    parser.add_argument("--log_interval", type=int, default=1, help="log interval")
    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
