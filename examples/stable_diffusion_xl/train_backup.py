import argparse
import ast
import os
import time
from functools import partial

from gm.data.loader import create_loader
from gm.helpers import (
    SD_XL_BASE_RATIOS,
    VERSION2SPECS,
    create_model,
    get_grad_reducer,
    get_loss_scaler,
    save_checkpoint,
    set_default,
)
from gm.util import get_obj_from_str, instantiate_from_config
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import Tensor


def get_parser_train():
    parser = argparse.ArgumentParser(description="train with sd-xl")
    parser.add_argument("--version", type=str, default="SDXL-base-1.0", choices=["SDXL-base-1.0", "SDXL-refiner-1.0"])
    parser.add_argument("--config", type=str, default="configs/training/sd_xl_base_finetune_lora.yaml")
    parser.add_argument(
        "--task",
        type=str,
        default="txt2img",
        choices=[
            "txt2img",
        ],
    )
    parser.add_argument("--weight", type=str, default="checkpoints/sd_xl_base_1.0_ms.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sd_xl_base_ratios", type=str, default="1.0")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="./runs")
    parser.add_argument("--log_interval", type=int, default=1, help="log interval")
    parser.add_argument("--save_ckpt_interval", type=int, default=1000, help="save ckpt interval")

    # args for infer
    parser.add_argument("--infer_during_train", type=ast.literal_eval, default=False)
    parser.add_argument("--infer_interval", type=int, default=1, help="log interval")

    # args for env
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument(
        "--ms_mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)"
    )
    parser.add_argument("--ms_amp_level", type=str, default="O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
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
    # Init Env
    args = set_default(args)

    # Create model
    config = OmegaConf.load(args.config)
    model, _ = create_model(
        config, checkpoints=args.weight.split(","), freeze=False, load_filter=False, amp_level=args.ms_amp_level
    )
    model.model.set_train(True)  # only unet

    # Create loader
    assert "data" in config
    dataloader = create_loader(data_path=args.data_path, rank=args.rank, rank_size=args.rank_size, **config.data)

    # Create train step func
    assert "optim" in config
    # get scheduler and lr
    base_lr = config.optim.get("base_learning_rate", 1.0e-4)
    if "scheduler_config" in config.optim:
        scheduler_config = config.optim.get("scheduler_config")
        scheduler = instantiate_from_config(scheduler_config)
        lr = [base_lr * scheduler(step) for step in range(config.data.total_step)]
    else:
        print(f"scheduler_config not exist, train with base_lr {base_lr}")
        lr = base_lr
    # get optimizer
    optimizer_config = config.optim.get("optimizer_config", {"target": "mindspore.nn.SGD"})
    optimizer = get_obj_from_str(optimizer_config["target"])(
        model.model.trainable_params(), learning_rate=lr, **optimizer_config.get("params", dict())
    )
    reducer = get_grad_reducer(is_parallel=False, parameters=optimizer.parameters)
    scaler = get_loss_scaler(ms_loss_scaler="static", scale_value=1024)
    train_step_fn = partial(
        model.train_step,
        grad_func=model.get_grad_func(optimizer, reducer, scaler, jit=True),
    )

    # Start Training
    if args.task == "txt2img":
        train_txt2img(
            args, train_step_fn, dataloader=dataloader, optimizer=optimizer, model=model  # for log lr  # for infer
        )
    elif args.task == "img2img":
        raise NotImplementedError
    else:
        raise ValueError(f"unknown mode {args.task}")


def train_txt2img(args, train_step_fn, dataloader, optimizer=None, model=None):  # for print  # for infer/ckpt
    dtype = ms.float32 if args.ms_amp_level not in ("O2", "O3") else ms.float16
    total_step = dataloader.get_dataset_size()
    loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    s_time = time.time()
    for i, data in enumerate(loader):
        # Data to tensor
        data = data["samples"]
        data = {k: (Tensor(v, dtype) if k != "txt" else v.tolist()) for k, v in data.items()}

        # Train a step
        if i == 0:
            print(
                "The first step will be compiled for the graph, which may take a long time; "
                "You can come back later :).",
                flush=True,
            )
        loss = train_step_fn(data)

        # Print meg
        if (i + 1) % args.log_interval == 0 and args.rank % 8 == 0:
            if optimizer.dynamic_lr:
                cur_lr = optimizer.learning_rate(Tensor(i, ms.int32)).asnumpy().item()
            else:
                cur_lr = optimizer.learning_rate.asnumpy().item()
            print(
                f"Step {i + 1}/{total_step}, size: {data['image'].shape[2:]}, lr: {cur_lr}, loss: {loss.asnumpy():.6f}"
                f", time cost: {(time.time()-s_time) * 1000 / args.log_interval:.2f} ms",
                flush=True,
            )
            s_time = time.time()

        # Save checkpoint
        if (i + 1) % args.save_ckpt_interval == 0 and args.rank % 8 == 0:
            model.model.set_train(False)  # only unet
            save_ckpt_dir = os.path.join(args.save_path, "weights", args.version + f"_{(i+1)}.ckpt")
            save_checkpoint(
                model,
                save_ckpt_dir,
                only_save_lora=False
                if not hasattr(model.model.diffusion_model, "only_save_lora")
                else model.model.diffusion_model.only_save_lora,
            )
            model.model.set_train(True)  # only unet

        # Infer during train
        if (i + 1) % args.infer_interval == 0 and args.infer_during_train:
            print(f"Step {i + 1}/{total_step}, infer starting...")
            infer_during_train(
                model=model,
                prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                save_path=os.path.join(args.save_path, "txt2img/", f"step_{i+1}_rank_{args.rank}"),
            )
            print(f"Step {i + 1}/{total_step}, infer done.", flush=True)


def infer_during_train(model, prompt, save_path):
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
    )
    perform_save_locally(save_path, out)


if __name__ == "__main__":
    parser = get_parser_train()
    args, _ = parser.parse_known_args()
    train(args)
