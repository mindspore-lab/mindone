import argparse
import logging
import os
import time

import yaml
from common import init_env
from ldm.data.dataset_textual_inversion import load_data
from ldm.modules.logger import set_logger
from ldm.modules.textual_inversion.manager import TextualInversionManager
from ldm.modules.train.ema import EMA
from ldm.modules.train.lr_schedule import create_scheduler
from ldm.modules.train.optim import build_optimizer
from ldm.modules.train.trainer import TrainOneStepWrapper
from ldm.util import count_params, instantiate_from_config, load_pretrained_model, str2bool
from utils.download import download_checkpoint

import mindspore as ms
from mindspore import Tensor
from mindspore.nn.wrap.loss_scale import FixedLossScaleUpdateCell

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)

_version_cfg = {
    "2.1": ("sd_v2-1_base-7c8d09ce.ckpt", "v2-inference.yaml", 512),
    "2.1-v": ("sd_v2-1_768_v-061732d1.ckpt", "v2-vpred-inference.yaml", 768),
    "2.0": ("sd_v2_base-57526ee4.ckpt", "v2-inference.yaml", 512),
    "2.0-v": ("sd_v2_768_v-e12e3a9b.ckpt", "v2-vpred-inference.yaml", 768),
    "1.5": ("sd_v1.5-d0ab7146.ckpt", "v1-inference.yaml", 512),
    "1.5-wukong": ("wukong-huahua-ms.ckpt", "v1-inference-chinese.yaml", 512),
}
_URL_PREFIX = "https://download.mindspore.cn/toolkits/mindone/stable_diffusion"
_MIN_CKPT_SIZE = 4.0 * 1e9


def read_template_file(template_file):
    assert os.path.exists(template_file), f"{template_file} does not exist!"
    with open(template_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


def parse_args():
    parser = argparse.ArgumentParser(description="A training script for dreambooth.")

    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument(
        "--jit_level",
        default="O2",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports ['O0', 'O1', 'O2']."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        nargs="?",
        default="1.5",
        help="Stable diffusion version. Options: '2.1', '2.1-v', '2.0', '2.0-v', '1.5', '1.5-wukong'",
    )
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="Enable parallel processing")
    parser.add_argument(
        "--train_config",
        default="configs/train/train_config_textual_inversion_v1.yaml",
        type=str,
        help="Specify the path to the train config file",
    )
    parser.add_argument(
        "--model_config", default="configs/v1-train-textual-inversion.yaml", type=str, help="model config path"
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Specify the path to the training data directory",
    )
    parser.add_argument(
        "--pretrained_model_path", default=None, type=str, help="Specify the directory of the pretrained model"
    )
    parser.add_argument(
        "--pretrained_model_file",
        default=None,
        type=str,
        help="Specify the filename of the pretrained model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output",
        help="Specify the output directory where the model predictions and checkpoints will be written.",
    )

    #
    parser.add_argument(
        "--learnable_property",
        type=str,
        default="object",
        choices=["object", "style"],
        help="Choose between 'object' and 'style'",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default=None,
        help="A token to use as initializer word." " If None, the new embedding will be initialized randomly",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--train_data_repeats",
        type=int,
        default=1,
        help=("Repeat the train(finetune) images by N times"),
    )
    # image
    parser.add_argument(
        "--random_crop",
        default=True,
        type=str2bool,
        help="Specify whether to use random crop. If set to False, center crop will be used.",
    )
    parser.add_argument("--filter_small_size", default=True, type=str2bool, help="filter small images")
    parser.add_argument("--image_size", default=512, type=int, help="Specify the size of images.")
    parser.add_argument("--image_filter_size", default=256, type=int, help="image filter size")
    parser.add_argument(
        "--replace_small_images",
        default=True,
        type=str2bool,
        help="replace the small-size images with other training samples",
    )
    parser.add_argument("--enable_modelarts", default=False, type=str2bool, help="run codes in ModelArts platform")
    parser.add_argument(
        "--train_batch_size", default=1, type=int, help="Specify the batch size (per device) for training."
    )
    parser.add_argument("--callback_size", default=1, type=int, help="Specify the callback size.")
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Specify the batch size (per device) for sampling images."
    )
    parser.add_argument("--overflow_still_update", type=str2bool, default=True)
    parser.add_argument(
        "--ckpt_save_interval", default=600, type=int, help="Save checkpoint every this number of steps."
    )
    parser.add_argument("--log_interval", default=1, type=int, help="Save checkpoint every this number of steps.")
    parser.add_argument(
        "--gradient_accumulation_steps", default=4, type=int, help="Specify the gradient accumulation steps."
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument(
        "--start_learning_rate", default=5e-5, type=float, help="Specify the initial learning rate for Adam."
    )
    parser.add_argument("--end_learning_rate", default=1e-7, type=float, help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument("--scheduler", default="multi_step_decay", type=str, help="scheduler.")
    parser.add_argument(
        "--milestones", default=[200, 400, 800, 1600, 3200], type=list, help="milestones for multi_step_decay"
    )
    parser.add_argument("--decay_rate", default=0.9, help="the decay rate for multi-step decay lr scheduler")
    parser.add_argument(
        "--scale_lr",
        default=False,
        type=str2bool,
        help="Specify whether to scale the learning rate based on the batch size, gradient accumulation steps, and n cards.",
    )
    parser.add_argument("--init_loss_scale", default=2048, type=float, help="Specify the initial loss scale.")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="Specify the loss scale factor.")
    parser.add_argument("--scale_window", default=200, type=float, help="Specify the scale window.")
    parser.add_argument("--use_ema", default=False, type=str2bool, help="Specify whether to use EMA.")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="Specify whether to apply gradient clipping.")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Specify the maximum gradient norm for clipping. This is effective when `clip_grad` is enabled.",
    )
    # optimizer
    parser.add_argument(
        "--optim", default="adamw", type=str, help="Specify the optimizer type. Options: ['adam', 'adamw']"
    )
    parser.add_argument(
        "--betas", type=float, default=[0.9, 0.999], help="Specify the [beta1, beta2] parameter for the Adam optimizer."
    )
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay.")
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="Specify the log level. Options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Specify a seed for reproducible training.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=6000,
        help="Specify the maximum number of steps to train. If set, args.epochs will not be applied.",
    )
    parser.add_argument(
        "--template_file",
        type=str,
        default=None,
        help=(
            "the template file which provides a list of strings of templates, like `a photo of {{}}`."
            " If `None`, it will use default templates for each `learnable_property`."
        ),
    )
    parser.add_argument("--num_workers", default=1, type=int, help="the number of modelarts workers")
    parser.add_argument(
        "--json_data_path",
        default="mindone/examples/stable_diffusion_v2/ldm/data/num_samples_64_part.json",
        type=str,
        help="the path of num_samples.json containing a dictionary with 64 parts. "
        "Each part is a large dictionary containing counts of samples of 533 tar packages.",
    )
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    default_args = parser.parse_args()
    if default_args.train_config:
        default_args.train_config = os.path.join(abs_path, default_args.train_config)
        with open(default_args.train_config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    args.model_config = os.path.join(abs_path, args.model_config)

    if not args.train_data_dir:
        raise ValueError("The training data directory must be specified.")
    if args.num_vectors < 1:
        raise ValueError(f"num_vectors has to be larger or equal to 1, but is {args.num_vectors}")
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError(f"max_steps has to be larger or equal to 1, but is {args.max_steps}")
    if args.version:
        if args.version not in _version_cfg:
            raise ValueError(f"Unknown version: {args.version}. Supported SD versions are: {list(_version_cfg.keys())}")
    if args.pretrained_model_path is None:
        args.pretrained_model_path = "models/"
    if args.pretrained_model_file is None:
        ckpt_name = _version_cfg[args.version][0]
        args.pretrained_model_file = ckpt_name
    ckpt_path = os.path.join(args.pretrained_model_path, args.pretrained_model_file)
    desire_size = _version_cfg[args.version][2]
    if args.image_size != desire_size:
        logger.warning(
            f"The optimal H, W for SD {args.version} is ({desire_size}, {desire_size}) . But got ({args.image_size})."
        )
    # download if not exists or not complete
    if os.path.exists(ckpt_path):
        if os.path.getsize(ckpt_path) < _MIN_CKPT_SIZE:
            logger.warning(
                f"WARNING: The checkpoint size is too small {args.ckpt_path}. Please check and remove it if it is incomplete!"
            )
    if not os.path.exists(ckpt_path):
        logger.info(f"Start downloading checkpoint {_version_cfg[args.version][0]} ...")
        ckpt_name = _version_cfg[args.version][0]
        download_checkpoint(os.path.join(_URL_PREFIX, ckpt_name), args.pretrained_model_path)
    logger.info(args)
    return args


def main(args):
    # init
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        jit_level=args.jit_level,
        enable_modelarts=args.enable_modelarts,
        num_workers=args.num_workers,
        json_data_path=args.json_data_path,
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))
    if args.scale_lr:
        args.start_learning_rate = (
            args.start_learning_rate * args.gradient_accumulation_steps * device_num * args.train_batch_size
        )
        logger.info(f"When scale_lr=True, the effective learning rate is {args.start_learning_rate}")

    # 1. Load SD
    latent_diffusion_with_loss = instantiate_from_config(args.model_config)
    pretrained_ckpt = os.path.join(args.pretrained_model_path, args.pretrained_model_file)
    load_pretrained_model(pretrained_ckpt, latent_diffusion_with_loss)
    latent_diffusion_with_loss.set_train(False)
    for _, p in latent_diffusion_with_loss.parameters_and_names():
        p.requires_grad = False

    # 2. Set Textual Inversion Manager to handle resize token embedding tables, etc.
    manager = TextualInversionManager(latent_diffusion_with_loss, args.placeholder_token, args.num_vectors)
    placeholder_tokens = manager.placeholder_tokens
    latent_diffusion_with_loss = manager.initiate_textual_inversion_params()
    tokenizer = manager.tokenizers[0]
    placeholder_tokens = manager.placeholder_tokens
    # build dataloader
    train_dataloader = load_data(
        args.train_data_dir,
        args.train_batch_size,
        tokenizer,
        args.train_data_repeats,
        learnable_property=args.learnable_property,
        templates=None if args.template_file is None else read_template_file(args.template_file),
        placeholder_token=(" ".join(placeholder_tokens)),
        image_size=args.image_size,
        image_filter_size=args.image_filter_size,
        device_num=device_num,
        rank_id=rank_id,
        random_crop=args.random_crop,
        filter_small_size=args.filter_small_size,
        replace=args.replace_small_images,
        enable_modelarts=args.enable_modelarts,
    )

    N_batches = len(train_dataloader)
    if args.max_steps is not None:
        args.epochs = args.max_steps // N_batches
        logger.info(f"max_steps is set to {args.max_steps}, epochs is changed to {args.epochs}")
    else:
        assert args.epochs is not None, "At least one of `max_steps` and `epochs` should be set."

    # all parameters in the ldm are frozen, except for the token embeddings in the text encoder.
    manager.set_train_textual_inversion(True)
    trainable_params = manager.get_textual_inversion_params()
    for p in trainable_params:
        p.requires_grad = True
    assert (
        len(latent_diffusion_with_loss.trainable_params()) == 1
    ), f"expect to train 1 parameter, but got {len(latent_diffusion_with_loss.trainable_params())} trainable params"

    # build learning rate scheduler
    dataset_size = train_dataloader.get_dataset_size()
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
        scheduler=args.scheduler,
        lr=args.start_learning_rate,
        min_lr=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        milestones=args.milestones,
        decay_rate=args.decay_rate,
        num_epochs=args.epochs,
    )

    optimizer = build_optimizer(
        model=latent_diffusion_with_loss,
        name=args.optim,
        betas=args.betas,
        weight_decay=args.weight_decay,
        lr=lr,
    )

    loss_scaler = FixedLossScaleUpdateCell(loss_scale_value=args.init_loss_scale)

    start_epoch = 0
    # trainer (standalone and distributed)
    ema = (
        EMA(
            latent_diffusion_with_loss.model,
            ema_decay=0.9999,
        )
        if args.use_ema
        else None
    )

    net_with_grads = TrainOneStepWrapper(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=not args.overflow_still_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    # log
    if rank_id == 0:
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        num_params, num_trainable_params = count_params(latent_diffusion_with_loss)
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"Model: StableDiffusion v{args.version}",
                f"Precision: {latent_diffusion_with_loss.model.diffusion_model.dtype}",
                f"Num params: {num_params}",
                f"Num trainable params: {num_trainable_params}",
                f"Learning rate: {args.start_learning_rate}",
                f"Weight decay: {args.weight_decay}",
                f"Batch size: {args.train_batch_size}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"Initial Loss Scale: {args.init_loss_scale}",
                f"Placeholder token: {args.placeholder_token}",
                f"Initializer token: {args.initializer_token}",
                f"Number of vectors: {args.num_vectors}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")

    train_txt2img(
        args,
        manager,
        net_with_grads,
        rank_id=rank_id,
        start_epoch=start_epoch,
        dataloader=train_dataloader,
        optimizer=optimizer,
    )


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
    args,
    manager,
    train_step_fn,
    dataloader,
    rank_id,
    start_epoch=0,
    optimizer=None,
):  # for print  # for infer/ckpt
    # 1. set training hyperparameters
    total_step = len(dataloader) * args.epochs
    text_encoders = manager.text_encoders
    # 2. get initial text embedding data, which is used to reset old text embeddings during training
    initial_text_embeds = [
        ms.Tensor(t.get_input_embeddings().value().asnumpy()[: -args.num_vectors]) for t in text_encoders
    ]

    # 3. training loop
    if args.mode == 0:
        logger.info(
            "The first step will compile the graph, which may take longer time; " "You can come back later :)",
        )
    for i_epoch in range(start_epoch, args.epochs):
        # 3.1 train one epoch
        train_one_epoch(
            i_epoch,
            args,
            manager,
            train_step_fn,
            dataloader,
            optimizer,
            initial_text_embeds,
            total_step,
            rank_id,
        )


def train_one_epoch(
    i_epoch,
    args,
    manager,
    train_step_fn,
    dataloader,
    optimizer,
    initial_text_embeds,
    total_step,
    rank_id,
):
    s_time = time.time()
    for i, data in enumerate(dataloader):
        manager.set_train_textual_inversion(True)
        i_step = i + i_epoch * len(dataloader) + 1
        image, tokens = data
        # Train a step
        loss, overflow, _ = train_step_fn(image, tokens)
        if overflow:
            if args.overflow_still_update:
                logger.info(f"Step {i_step}/{total_step}, overflow, still update.")
            else:
                logger.info(f"Step {i_step}/{total_step}, overflow, skip.")

        # textual_inversion trainable parameters set_train(False) temporarily for logging purpose
        manager.set_train_textual_inversion(False)

        # reset the old text embedding table to its original value
        text_encoders = manager.text_encoders
        text_embedding_tables = [t.get_input_embeddings() for t in text_encoders]
        i_text_encoder = 0

        for text_embedding_table, initial_text_embedding_data in zip(text_embedding_tables, initial_text_embeds):
            replace_text_embeds(text_embedding_table, initial_text_embedding_data, i_text_encoder, verbose=False)
            i_text_encoder += 1

        # Print meg
        if i_step % args.log_interval == 0 and rank_id % 8 == 0:
            if optimizer.dynamic_lr:
                cur_lr = optimizer.learning_rate(Tensor(i_step - 1, ms.int32)).asnumpy().item()
            else:
                cur_lr = optimizer.learning_rate.asnumpy().item()
            logger.info(
                f"Step {i_step}/{total_step}, lr: {cur_lr}, loss: {loss.asnumpy():.6f}"
                f", time cost: {(time.time()-s_time) * 1000 / args.log_interval:.2f} ms",
            )
            s_time = time.time()

        # Save checkpoint
        if i_step % args.ckpt_save_interval == 0 and rank_id % 8 == 0:
            save_ckpt_dir = os.path.join(args.output_path, "weights")
            if not os.path.exists(save_ckpt_dir):
                os.makedirs(save_ckpt_dir)
            save_filename = "SDv" + args.version + f"_textual_inversion_{i_step}.ckpt"
            manager.save_checkpoint_textual_inversion(
                os.path.join(save_ckpt_dir, save_filename), args.num_vectors, placeholder_token=args.placeholder_token
            )


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
