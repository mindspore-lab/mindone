import argparse
import ast
import os
import time

from gm.data.loader import create_loader
from gm.helpers import (
    SD_XL_BASE_RATIOS,
    VERSION2SPECS,
    create_model,
    get_grad_reducer,
    get_learning_rate,
    get_loss_scaler,
    get_optimizer,
    set_default,
)
from gm.modules.textual_inversion.manager import TextualInversionManager
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import Tensor


def get_parser_train():
    parser = argparse.ArgumentParser(description="train textual inversion with sd-xl")
    parser.add_argument("--version", type=str, default="SDXL-base-1.0", choices=["SDXL-base-1.0", "SDXL-refiner-1.0"])
    parser.add_argument("--config", type=str, default="configs/training/sd_xl_base_finetune_textual_inversion.yaml")
    parser.add_argument(
        "--task",
        type=str,
        default="txt2img",
        choices=[
            "txt2img",
        ],
    )

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--clip_grad", default=True, type=ast.literal_eval, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--weight", type=str, default="checkpoints/sd_xl_base_1.0_ms.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sd_xl_base_ratios", type=str, default="1.0")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="./runs")
    parser.add_argument("--save_path_with_time", type=ast.literal_eval, default=True)
    parser.add_argument("--log_interval", type=int, default=1, help="log interval")
    parser.add_argument("--save_ckpt_interval", type=int, default=500, help="save ckpt interval")
    parser.add_argument("--data_sink", type=ast.literal_eval, default=False)
    parser.add_argument("--sink_size", type=int, default=1000)
    # args for infer
    parser.add_argument("--infer_during_train", type=ast.literal_eval, default=False)
    parser.add_argument(
        "--guidance_scale",
        type=int,
        default=7.0,
        help="the guidance scale for inference, only applies if infer_during_train is True.",
    )
    parser.add_argument("--infer_interval", type=int, default=500, help="inference interval")

    # args for env
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)"
    )
    parser.add_argument("--ms_amp_level", type=str, default="O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
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
    parser.add_argument(
        "--total_step",
        type=int,
        default=None,
        help="the number of training steps. " "If not provided, will use the `total_step` in training yaml file.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="The batch size for training. If not provided, will use `per_batch_size` in training yaml file.",
    )
    # arguments for textual inverision
    parser.add_argument(
        "--learnable_property",
        type=str,
        default="object",
        help="the learnable property of the concepts, choices: ['style', 'object', 'face]",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="<cat-toy>",
        help="the learnable property of the concepts, choices: ['style', 'object', 'face]",
    )
    parser.add_argument("--num_vectors", type=int, default=1, help="the number of newly learnable text embeddings")
    parser.add_argument(
        "--initializer_token",
        type=str,
        default=None,
        help=(
            "the token whose embedding is used to initialize the newly added token's embedding."
            " If `None`, the newly added token's embedding will be randomly initialized."
        ),
    )
    parser.add_argument(
        "--template_file",
        type=str,
        default=None,
        help=(
            "the template file which provides a list of strings of templates, like `a photo of {{}}`."
            " If `None`, it will use default templates for each learnable_property."
        ),
    )

    return parser


def read_template_file(template_file):
    assert os.path.exists(template_file), f"{template_file} does not exist!"
    with open(template_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def train(args):
    # Init Env
    args = set_default(args)
    assert (
        not args.data_sink
    ), "Textual Inversion does not support data sink mode because of repeated weight replacement during training."

    # 1. Create LDM Engine
    config = OmegaConf.load(args.config)
    s_time = time.time()
    model, _ = create_model(
        config,
        checkpoints=args.weight,
        freeze=False,
        load_filter=False,
        param_fp16=args.param_fp16,
        amp_level=args.ms_amp_level,
    )
    print(f"Model Initialization and Loading time : {(time.time()-s_time) * 1000} ms")
    model.set_train(False)
    for _, p in model.parameters_and_names():
        p.requires_grad = False

    # 2. Set Textual Inversion Manager to handle resize token embedding tables, etc.
    manager = TextualInversionManager(model, args.placeholder_token, args.num_vectors)
    placeholder_tokens = manager.placeholder_tokens
    model = manager.initiate_textual_inversion_params()
    # 3. Create loader
    assert "data" in config and "dataset_config" in config.data and "params" in config.data.dataset_config
    if args.total_step is not None:
        config.data["total_step"] = args.total_step
    if args.train_batch_size is not None:
        config.data["per_batch_size"] = args.train_batch_size
    total_step = config.data["total_step"]
    base_lr = config.optim["base_learning_rate"]
    train_batch_size = config.data["per_batch_size"]
    config.data.dataset_config.params["learnable_property"] = args.learnable_property
    config.data.dataset_config.params["placeholder_token"] = " ".join(placeholder_tokens)
    config.data.dataset_config.params["templates"] = (
        None if args.template_file is None else read_template_file(args.template_file)
    )
    dataloader = create_loader(data_path=args.data_path, rank=args.rank, rank_size=args.rank_size, **config.data)

    # 4. Create train step func
    assert "optim" in config
    lr = get_learning_rate(config.optim, config.data.total_step)
    # get optimizer
    manager.set_train_textual_inversion(True)
    trainable_params = manager.get_textual_inversion_params()
    for p in trainable_params:
        p.requires_grad = True
    assert (
        len(model.trainable_params()) == 2
    ), f"expect to train 2 parameters, but got {len(model.trainable_params())} trainable params"
    scaler = get_loss_scaler(ms_loss_scaler="static", scale_value=1024)

    optimizer = get_optimizer(config.optim, lr, params=model.trainable_params())
    reducer = get_grad_reducer(is_parallel=args.is_parallel, parameters=optimizer.parameters)

    from gm.models.trainer_factory import TrainOneStepCell

    train_step_fn = TrainOneStepCell(
        model,
        optimizer,
        reducer,
        scaler,
        overflow_still_update=args.overflow_still_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
    )
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
            f"Task type: {args.task}",
            f"AMP level: {args.ms_amp_level}",
            f"Base Learning rate: {base_lr}",
            f"Batch size: {train_batch_size}",
            f"Grad accumulation steps: {args.gradient_accumulation_steps}",
            f"Num of steps: {total_step}",
            f"Grad clipping: {args.clip_grad}",
            f"Max grad norm: {args.max_grad_norm}",
        ]
    )
    key_info += "\n" + "=" * 50
    print(key_info)

    print("Start training...")

    if args.task == "txt2img":
        train_txt2img(
            args,
            manager,
            train_step_fn,
            dataloader=dataloader,
            optimizer=optimizer,
            model=model,  # for log lr  # for infer
            placeholder_tokens=placeholder_tokens,
        )
    elif args.task == "img2img":
        raise NotImplementedError
    else:
        raise ValueError(f"unknown mode {args.task}")


def replace_text_embeds(t, i, i_text_encoder=0, verbose=True):
    assert i.shape[0] < t.shape[0]
    num_no_updates = i.shape[0]
    # check if the text embedding has been updated from the last training step
    if ms.ops.Equal()(t[:num_no_updates], i).all() and verbose:
        print("WARNING: No updates from the initial text embeds! This means the last update failed")
    data_to_copy = ms.ops.concat([i, t[num_no_updates:].value()], axis=0)
    ms.ops.Assign()(t, data_to_copy)
    if verbose:
        print(
            f"Newly learned text embedding {i_text_encoder}: min {t[num_no_updates:].min()}, max {t[num_no_updates:].max()}, mean {t[num_no_updates:].mean()}"
        )


def train_txt2img(
    args, manager, train_step_fn, dataloader, optimizer=None, model=None, placeholder_tokens=None, loss_scaler=None
):  # for print  # for infer/ckpt
    # 1. set training hyperparameters
    dtype = ms.float32 if args.ms_amp_level not in ("O2", "O3") else ms.float16
    total_step = dataloader.get_dataset_size()
    loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    s_time = time.time()
    text_encoders = manager.text_encoders
    if loss_scaler is not None:
        last_loss_scale = loss_scaler.scale_value.value()
    # 2. get initial text embedding data, which is used to reset old text embeddings during training
    initial_text_embeds = [
        ms.Tensor(t.get_input_embeddings().embedding_table.value().asnumpy()[: -args.num_vectors])
        for t in text_encoders
    ]

    # 3. training loop
    for i, data in enumerate(loader):
        manager.set_train_textual_inversion(True)
        # Get data, to tensor
        data = data["samples"]
        data = {k: (Tensor(v, dtype) if k != "txt" else v.tolist()) for k, v in data.items()}

        # Get image and tokens
        image = data[model.input_key]
        tokens, _ = model.conditioner.tokenize(data)
        tokens = [Tensor(t) for t in tokens]

        # Train a step
        if i == 0 and args.ms_mode == 0:
            print(
                "The first step will compile the graph, which may take longer time; " "You can come back later :)",
                flush=True,
            )
        loss, overflow = train_step_fn(image, *tokens)
        if overflow:
            if args.overflow_still_update:
                print(f"Step {i + 1}/{total_step}, overflow, still update.")
            else:
                print(f"Step {i + 1}/{total_step}, overflow, skip.")

        # textual_inversion trainable parameters set_train(False) temporarily for logging purpose
        manager.set_train_textual_inversion(False)

        # reset the old text embedding table to its original value
        text_encoders = manager.text_encoders
        text_embedding_tables = [t.get_input_embeddings().embedding_table for t in text_encoders]
        i_text_encoder = 0

        for text_embedding_table, initial_text_embedding_data in zip(text_embedding_tables, initial_text_embeds):
            replace_text_embeds(text_embedding_table, initial_text_embedding_data, i_text_encoder, verbose=False)
            i_text_encoder += 1

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
            save_ckpt_dir = os.path.join(args.save_path, "weights", args.version + f"_{(i+1)}.ckpt")
            manager.save_checkpoint_textual_inversion(
                save_ckpt_dir, args.num_vectors, placeholder_token=args.placeholder_token
            )

        # Infer during train
        if (i + 1) % args.infer_interval == 0 and args.infer_during_train:
            print(f"Step {i + 1}/{total_step}, infer starting...")
            infer_during_train(
                model=model,
                prompt="A photo of a {}".format(" ".join(placeholder_tokens)),
                save_path=os.path.join(args.save_path, "txt2img/", f"step_{i+1}_rank_{args.rank}"),
                amp_level=args.ms_amp_level,
                guidance_scale=args.guidance_scale,
            )
            print(f"Step {i + 1}/{total_step}, infer done.", flush=True)

        # monitor the loss scale change
        if loss_scaler is not None and args.rank % 8 == 0:
            if loss_scaler.scale_value.value() != last_loss_scale:
                print(f"Loss scale is updated from {last_loss_scale} to {loss_scaler.scale_value.value()}")
                last_loss_scale = loss_scaler.scale_value.value()


def infer_during_train(model, prompt, save_path, num_cols=1, amp_level="O2", guidance_scale=5.0):
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
    sampler, num_rows, num_cols = init_sampling(
        steps=40,
        num_cols=num_cols,
        guidance_scale=guidance_scale,
    )

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
        amp_level=amp_level,
    )
    perform_save_locally(save_path, out)


if __name__ == "__main__":
    parser = get_parser_train()
    args, _ = parser.parse_known_args()
    train(args)
