import logging
import os
import sys
from typing import Literal, Optional

from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import Path_fr, path_type

from mindspore import Model, amp, nn
from mindspore.train.callback import TimeMonitor

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from moviegen.dataset import ImageVideoDataset
from moviegen.models.llama import LlamaModel
from moviegen.pipelines import DiffusionWithLoss
from moviegen.schedulers import RFlowLossWrapper
from moviegen.utils import EMA, MODEL_DTYPE, MODEL_SPEC, load_ckpt_params

from mindone.data import create_dataloader
from mindone.trainers import create_optimizer, create_scheduler
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor
from mindone.trainers.zero import prepare_train_network
from mindone.utils import count_params, init_train_env, set_logger

# TODO: remove when VAE is added to the project
sys.path.append(os.path.join(__dir__, "../opensora_hpcai/"))
from opensora.models.vae.vae import OpenSoraVAE_V1_2

logger = logging.getLogger(__name__)

Path_dcc = path_type("dcc")  # path to a directory that can be created if it does not exist


def init_model(
    name: Literal["llama-1B", "llama-5B", "llama-30B"],
    pretrained_model_path: Optional[Path_fr] = None,
    enable_flash_attention: bool = True,
    recompute: bool = False,
    dtype: Literal["fp32", "fp16", "bf16"] = "fp32",
) -> LlamaModel:
    attn_implementation = "flash_attention" if enable_flash_attention else "eager"
    model = MODEL_SPEC[name](
        in_channels=4,
        out_channels=8,
        attn_implementation=attn_implementation,
        gradient_checkpointing=recompute,
        dtype=MODEL_DTYPE[dtype],
    )
    if pretrained_model_path:
        model = load_ckpt_params(model, pretrained_model_path)
    else:
        logger.info("Initialize network randomly.")
    return model


def main(args):
    # 1. init env
    args.train.output_path = os.path.join(__dir__, args.train.output_path.relative)
    os.makedirs(args.train.output_path, exist_ok=True)
    device_id, rank_id, device_num = init_train_env(**args.env)
    set_logger("", output_dir=args.train.output_path, rank=rank_id)

    # instantiate classes only after initializing training environment
    initializer = parser.instantiate_classes(cfg)

    # 2. model initialize and weight loading
    # 2.1 Llama 3
    network = init_model(**args.model)

    # 2.2 VAE
    logger.info("vae init")
    # TODO: add support of training with latents
    vae_args = args.vae.as_dict()
    vae_dtype = vae_args.pop("dtype")
    vae = OpenSoraVAE_V1_2(**vae_args).set_train(False)
    if vae_dtype != "fp32":
        vae_dtype = MODEL_DTYPE[vae_dtype]
        # FIXME: remove AMP and add custom dtype conversion support for better compatibility with PyNative
        amp.custom_mixed_precision(vae, black_list=amp.get_black_list() + [nn.GroupNorm], dtype=vae_dtype)

    # 2.4 LossWrapper
    rflow_loss_wrapper = RFlowLossWrapper(network)

    # 3. build training network
    latent_diffusion_with_loss = DiffusionWithLoss(rflow_loss_wrapper, vae)

    # 4. build dataset
    dataset = ImageVideoDataset(**args.dataset)
    transforms = (
        dataset.train_transforms(args.dataset.target_size) if not args.dataset.apply_transforms_dataset else None
    )
    dataloader = create_dataloader(
        dataset, transforms=transforms, device_num=device_num, rank_id=rank_id, **args.dataloader
    )

    # 5. build training utils: lr, optim, callbacks, trainer
    # 5.1 LR
    lr = create_scheduler(steps_per_epoch=dataloader.get_dataset_size(), **args.train.lr_scheduler)

    # 5.2 optimizer
    optimizer = create_optimizer(latent_diffusion_with_loss.trainable_params(), lr=lr, **args.train.optimizer)

    loss_scaler = initializer.train.loss_scaler

    # 5.3 trainer (standalone and distributed)
    ema = EMA(latent_diffusion_with_loss.network, **args.train.ema.init_args) if args.train.ema else None
    net_with_grads = prepare_train_network(
        latent_diffusion_with_loss, optimizer=optimizer, scale_sense=loss_scaler, ema=ema, **args.train.settings
    )

    model = Model(net_with_grads)

    # 5.4 callbacks
    callbacks = [OverflowMonitor()]

    if rank_id == 0:
        callbacks.extend(
            [
                TimeMonitor(args.train.save.log_interval),
                EvalSaveCallback(
                    network=latent_diffusion_with_loss.network,
                    model_name=args.model.name,
                    rank_id=rank_id,
                    ckpt_save_dir=os.path.join(args.train.output_path, "ckpt"),
                    ema=ema,
                    **args.train.save,
                ),
            ]
        )
        num_params_vae, num_params_trainable_vae = count_params(vae)
        num_params_network, num_params_trainable_network = count_params(network)
        num_params = num_params_vae + num_params_network
        num_params_trainable = num_params_trainable_vae + num_params_trainable_network
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.env.mode}",
                f"JIT level: {args.env.jit_level}",
                f"Distributed mode: {args.env.distributed}",
                f"Data path: {args.dataset.csv_path}",
                f"Number of samples: {len(dataset)}",
                f"Num params: {num_params:,} (network: {num_params_network:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"Model dtype: {args.model.dtype}",
                f"VAE dtype: {args.vae.dtype}",
                f"Learning rate: {args.train.lr_scheduler.lr:.0e}",
                f"Batch size: {args.dataloader.batch_size}",
                f"Image size: {args.dataset.target_size}",
                f"Frames: {args.dataset.sample_n_frames}",
                f"Weight decay: {args.train.optimizer.weight_decay}",
                f"Grad accumulation steps: {args.train.settings.gradient_accumulation_steps}",
                f"Num epochs: {args.train.epochs}",
                f"Loss scaler: {args.train.loss_scaler.class_path}",
                f"Init loss scale: {args.train.loss_scaler.init_args.loss_scale_value}",
                f"Grad clipping: {args.train.settings.clip_grad}",
                f"Max grad norm: {args.train.settings.clip_norm}",
                f"EMA: {ema is not None}",
                f"Enable flash attention: {args.model.enable_flash_attention}",
            ]
        )
        key_info += "\n" + "=" * 50
        print(key_info)
        parser.save(args, args.train.output_path + "/config.yaml", format="yaml", overwrite=True)

    # 6. train
    logger.info("Start training...")
    model.train(args.train.epochs, dataloader, callbacks=callbacks)


if __name__ == "__main__":
    parser = ArgumentParser(description="Movie Gen training script.")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to load a config yaml file that describes the setting which will override the default arguments.",
    )
    parser.add_function_arguments(init_train_env, "env")
    parser.add_function_arguments(init_model, "model")
    parser.add_function_arguments(OpenSoraVAE_V1_2, "vae", fail_untyped=False)
    parser.add_argument(
        "--vae.dtype", default="fp32", type=str, choices=["fp32", "fp16", "bf16"], help="VAE model precision."
    )
    parser.add_class_arguments(
        ImageVideoDataset, "dataset", skip={"frames_mask_generator", "t_compress_func"}, instantiate=False
    )
    parser.add_function_arguments(
        create_dataloader, "dataloader", skip={"dataset", "transforms", "device_num", "rank_id"}
    )
    parser.link_arguments("env.debug", "dataloader.debug", apply_on="parse")
    parser.add_function_arguments(create_scheduler, "train.lr_scheduler", skip={"steps_per_epoch"})
    parser.add_function_arguments(create_optimizer, "train.optimizer", skip={"params", "lr"})
    parser.add_subclass_arguments(
        nn.Cell,
        "train.loss_scaler",
        fail_untyped=False,  # no typing in mindspore
        help="mindspore.nn.FixedLossScaleUpdateCell or mindspore.nn.DynamicLossScaleUpdateCell",
    )
    parser.add_function_arguments(
        prepare_train_network, "train.settings", skip={"network", "optimizer", "scale_sense", "ema"}
    )
    parser.add_subclass_arguments(EMA, "train.ema", skip={"network"}, required=False, instantiate=False)
    parser.add_argument(
        "--train.output_path", default="output/", type=Path_dcc, help="Output directory to save training results."
    )
    parser.add_argument("--train.epochs", default=10, type=int, help="Number of epochs to train. Default: 100.")
    parser.link_arguments("train.epochs", "train.lr_scheduler.num_epochs", apply_on="parse")
    parser.add_class_arguments(
        EvalSaveCallback,
        "train.save",
        skip={"network", "rank_id", "ckpt_save_dir", "output_dir", "ema", "start_epoch", "model_name"},
        instantiate=False,
    )
    cfg = parser.parse_args()
    main(cfg)
