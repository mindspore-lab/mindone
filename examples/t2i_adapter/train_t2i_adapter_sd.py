import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from adapters import get_adapter
from data.dataset_with_cond import CondDataset
from jsonargparse import ActionConfigFile, ArgumentParser
from omegaconf import OmegaConf
from pipelines.sd_pipeline import SDAdapterPipeline

import mindspore as ms
from mindspore import Model, nn
from mindspore.train.callback import LossMonitor, TimeMonitor

sys.path.append("../../")  # FIXME: remove in future when mindone is ready for install
from mindone.data import create_dataloader
from mindone.utils.env import init_train_env

sys.path.append("../stable_diffusion_v2/")
from ldm.data.transforms import TokenizerWrapper
from ldm.modules.logger import set_logger
from ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor
from ldm.modules.train.optim import build_optimizer
from ldm.modules.train.trainer import TrainOneStepWrapper
from ldm.util import count_params
from text_to_image import load_model_from_config


def main(args, initializer):
    # step 1: initialize environment
    logger = logging.getLogger(__name__)
    device_id, rank_id, device_num = init_train_env(**args.environment)
    if args.environment.mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": args.jit_level})
    output_dir = Path(args.train.output_dir) / args.adapter.condition / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_logger(output_dir=str(output_dir), rank=rank_id)

    # step 2: load SD model
    sd_config = OmegaConf.load(args.sd_config)
    sd_model = load_model_from_config(sd_config, args.sd_ckpt)

    # step 3: prepare train dataset and dataloader
    dataset = initializer.train.dataset
    transforms = dataset.train_transforms(args.adapter.condition, TokenizerWrapper(sd_model.cond_stage_model.tokenizer))
    train_dataloader = create_dataloader(
        dataset,
        transforms=transforms,
        device_num=device_num,
        rank_id=rank_id,
        debug=args.environment.debug,
        enable_modelarts=args.environment.enable_modelarts,
        **args.train.dataloader,
    )

    # step 4: load Adapter
    adapter_model = get_adapter("sd", **args.adapter, train=True)
    full_model = SDAdapterPipeline(sd_model, adapter_model)

    # step 5: create optimizer and train the same way as regular SD
    optimizer = build_optimizer(adapter_model, **args.train.optimizer)

    loss_scaler = nn.DynamicLossScaleUpdateCell(**args.LossScale)

    net_with_grads = TrainOneStepWrapper(
        full_model, optimizer=optimizer, scale_sense=loss_scaler, **args.train.settings
    )

    callbacks = [OverflowMonitor()]

    if rank_id == 0:
        callbacks.extend(
            [
                TimeMonitor(1),
                LossMonitor(),
                EvalSaveCallback(
                    network=adapter_model,
                    model_name="t2iadapter",
                    rank_id=rank_id,
                    ckpt_save_dir=str(output_dir / "ckpt"),
                    ckpt_save_policy="latest_k",
                    ckpt_max_keep=10,
                ),
            ]
        )

        num_params_unet, _ = count_params(sd_model.model.diffusion_model)
        num_params_text_encoder, _ = count_params(sd_model.cond_stage_model)
        num_params_vae, _ = count_params(sd_model.first_stage_model)
        num_params, _ = count_params(sd_model)
        num_params_adapter, num_trainable_params = count_params(adapter_model)

        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"Debugging: {args.environment.debug}",
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.environment.mode}",
                f"JIT level: {args.jit_level}",
                f"Distributed mode: {args.environment.distributed}",
                "Model: StableDiffusion v2.1",  # Support 1.x
                f"Num params SD: {num_params:,} (unet: {num_params_unet:,}, text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num params Adapter: {num_params_adapter:,} (trainable: {num_trainable_params:,})",
                f"Precision SD: {sd_model.model.diffusion_model.dtype}",
                f"Precision Adapter: {'Float16' if args.adapter.use_fp16 else 'Float32'}",
                f"Num epochs: {args.train.epochs}",
                f"Learning rate: {args.train.optimizer.lr}",
                f"Batch size: {args.train.dataloader.batch_size}",
                f"Weight decay: {args.train.optimizer.weight_decay}",
                f"Grad accumulation steps: {args.train.settings.gradient_accumulation_steps}",
                f"Grad clipping: {args.train.settings.clip_grad}",
                f"Max grad norm: {args.train.settings.clip_norm}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")
        # backup config files
        shutil.copyfile(args.sd_config, output_dir / "sd_config.yaml")  # SD's parameters are not modified
        ArgumentParser().save(args, output_dir / "adapter_config.yaml", format="yaml", skip_check=True)

    model = Model(net_with_grads)
    model.train(
        args.train.epochs,
        train_dataloader,
        callbacks=callbacks,
        dataset_sink_mode=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(init_train_env, "environment")
    parser.add_argument("--train.epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument(
        "--train.output_dir",
        type=str,
        default="output/t2i_adapter_v2.1/",
        help="Output directory for saving training results.",
    )
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports ['O0', 'O1', 'O2']."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_subclass_arguments(CondDataset, "train.dataset")
    parser.add_function_arguments(
        create_dataloader,
        "train.dataloader",
        skip={"dataset", "transforms", "batch_transforms", "device_num", "rank_id", "debug", "enable_modelarts"},
    )
    parser.add_function_arguments(build_optimizer, "train.optimizer", skip={"model"})
    parser.add_class_arguments(
        TrainOneStepWrapper, "train.settings", skip={"network", "optimizer", "scale_sense", "ema"}, instantiate=False
    )
    parser.add_argument("--sd_config", type=str, required=True)
    parser.add_argument("--sd_ckpt", type=str, required=True)
    parser.add_function_arguments(get_adapter, "adapter", skip={"diffusion_model", "train"})
    parser.add_class_arguments(nn.DynamicLossScaleUpdateCell, "LossScale", instantiate=False, fail_untyped=False)

    cfg = parser.parse_args()
    cfg.pop("config")  # not needed anymore after instantiation
    init = parser.instantiate_classes(cfg)

    main(cfg, init)
