import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import yaml
from ldm.data.dataset_db import load_data
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.logger import set_logger
from ldm.modules.lora import inject_trainable_lora, inject_trainable_lora_to_textencoder
from ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor
from ldm.modules.train.ema import EMA
from ldm.modules.train.optim import build_optimizer
from ldm.modules.train.parallel_config import ParallelConfig
from ldm.modules.train.tools import set_random_seed
from ldm.modules.train.trainer import TrainOneStepWrapper
from ldm.util import instantiate_from_config, load_pretrained_model, str2bool
from omegaconf import OmegaConf
from PIL import Image

import mindspore as ms
from mindspore import Model, context
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import LossMonitor, TimeMonitor

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)


def init_env(args):
    set_random_seed(args.seed)
    ms.set_context(max_device_memory=args.max_device_memory)  # TODO: why limit?
    ms.set_context(mode=args.mode)  # needed for MS2.0
    if args.use_parallel:
        init()
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = get_group_size()
        ParallelConfig.dp = device_num
        rank_id = get_rank()
        args.rank = rank_id
        logger.debug("Device_id: {}, rank_id: {}, device_num: {}".format(device_id, rank_id, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL,
            # parallel_mode=context.ParallelMode.AUTO_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        device_num = 1
        device_id = int(os.getenv("DEVICE_ID", 0))
        rank_id = 0
        args.rank = rank_id

    context.set_context(
        mode=args.mode,
        device_target="Ascend",
        device_id=device_id,
        pynative_synchronize=False,  # for debug in pynative mode
    )

    return rank_id, device_id, device_num


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


def parse_args():
    parser = argparse.ArgumentParser(description="A training script for dreambooth.")

    parser.add_argument(
        "--train_config",
        default="",
        type=str,
        help="train config path to load a yaml file that override the default arguments",
    )
    parser.add_argument("--unet_initialize_random", default=False, type=str2bool, help="initialize unet randomly")
    parser.add_argument("--dataset_sink_mode", default=False, type=str2bool, help="sink mode")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="Enable parallel processing")
    parser.add_argument("--max_device_memory", type=str, default="30GB", help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument("--use_lora", default=False, type=str2bool, help="Enable LoRA finetuning")
    parser.add_argument("--lora_ft_unet", default=False, type=str2bool, help="whether to apply lora finetune to unet")
    parser.add_argument(
        "--lora_ft_text_encoder", default=False, type=str2bool, help="whether to apply lora finetune to text encoder"
    )
    parser.add_argument("--lora_fp16", default=True, type=str2bool, help="Specify whether to use fp16 for LoRA params.")
    parser.add_argument(
        "--lora_rank",
        default=4,
        type=int,
        help="Specify the rank of LoRA. A higher rank results in a larger LoRA model and"
        "potentially better generation quality.",
    )
    parser.add_argument(
        "--model_config", default="configs/train_dreambooth_sd_v2.yaml", type=str, help="model config path"
    )
    parser.add_argument(
        "--pretrained_model_path", default="", type=str, help="Specify the pretrained model from this checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output",
        help="Specify the output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="Specify the folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="Specify the folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="Specify the prompt with an identifier that specifies the instance.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="Specify the prompt to identify images in the same class as the provided instance images.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
        "Simplified: `uc + scale * (uc - prompt)`",
    )

    # loss
    parser.add_argument(
        "--with_prior_preservation", type=str2bool, default=True, help="Specify whether to use prior preservation loss."
    )
    parser.add_argument(
        "--prior_loss_weight", type=float, default=1.0, help="Specify the weight of the prior preservation loss."
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Specify the number of class images for prior preservation loss. If there are not enough images"
            " already present in class_data_dir, additional images will be sampled using class_prompt."
        ),
    )
    parser.add_argument(
        "--train_data_repeats",
        type=int,
        default=40,
        help=(
            "Repeat the instance images by N times in order to match the number of class images."
            " We recommend setting it as [number of class images] / [number of instance images]."
        ),
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=50,
        help="Specify the number of ddim sampling steps.",
    )
    # image
    parser.add_argument(
        "--random_crop",
        default=False,
        type=str2bool,
        help="Specify whether to use random crop. If set to False, center crop will be used.",
    )
    parser.add_argument("--image_size", default=512, type=int, help="Specify the size of images.")
    parser.add_argument(
        "--train_text_encoder",
        type=str2bool,
        default=True,
        help="Specify whether to train the text encoder. If set, the text encoder will be trained.",
    )
    parser.add_argument(
        "--train_batch_size", default=2, type=int, help="Specify the batch size (per device) for training."
    )
    parser.add_argument("--callback_size", default=1, type=int, help="Specify the callback size.")
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Specify the batch size (per device) for sampling images."
    )
    parser.add_argument(
        "--ckpt_save_interval", default=1, type=int, help="Save checkpoint every this number of epochs."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", default=1, type=int, help="Specify the gradient accumulation steps."
    )
    parser.add_argument("--warmup_steps", default=200, type=int, help="Specify the number of warmup steps.")
    parser.add_argument(
        "--start_learning_rate", default=5e-6, type=float, help="Specify the initial learning rate for Adam."
    )
    parser.add_argument("--end_learning_rate", type=float, help="Specify the end learning rate for the optimizer.")
    parser.add_argument(
        "--decay_steps", default=0, type=int, help="Specify the number of decay steps for the learning rate."
    )
    parser.add_argument("--init_loss_scale", default=512, type=float, help="Specify the initial loss scale.")
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
    parser.add_argument("--epochs", type=int, default=8)

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

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            logger.warning("With with_prior_preservation=False, class_data_dir will not be used.")
        if args.class_prompt is not None:
            logger.warning("With with_prior_preservation=False, class_prompt will not be used.")

    if args.end_learning_rate and args.start_learning_rate < args.end_learning_rate:
        raise ValueError(
            f"The start learning rate {args.start_learning_rate} must be no less"
            " than the end learning rate {args.end_learning_rate}."
        )
    if args.lora_ft_text_encoder or args.lora_ft_unet:
        assert args.use_lora, "Lora has to be True when `lora_ft_text_encoder` or `lora_ft_unet` is True"
    if args.use_lora and args.lora_ft_text_encoder:
        if not args.train_text_encoder:
            raise ValueError("When `lora_ft_text_encoder` is True, `train_text_encoder` has to be True")

    logger.info(args)
    return args


def generate_class_images(args):
    """Generate images for the class"""
    class_images_dir = Path(args.class_data_dir)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))
    if cur_class_images >= args.num_class_images:
        logger.info(f"Found {cur_class_images} class images. No need to generate more class images.")
        return None
    logger.info("Start generating class images. ")
    model = instantiate_from_config(args.model_config)
    load_pretrained_model(args.pretrained_model_path, model)
    model.set_train(False)
    for param in model.get_parameters():
        param.requires_grad = False
    sampler = DDIMSampler(model)
    if cur_class_images < args.num_class_images:
        num_new_images = args.num_class_images - cur_class_images
        logger.info(f"Number of class images to sample: {num_new_images}.")
        N_prompts = num_new_images // args.sample_batch_size
        sample_dataset = [args.class_prompt] * (
            N_prompts + 1 if num_new_images % args.sample_batch_size != 0 else N_prompts
        )
    start_time = time.time()
    start_code = None
    for prompt in sample_dataset:
        uc_prompts = args.sample_batch_size * [""]
        c_prompts = args.sample_batch_size * [prompt]
        uc = model.get_learned_conditioning(model.tokenize(uc_prompts))
        c = model.get_learned_conditioning(model.tokenize(c_prompts))
        shape = [4, args.image_size // 8, args.image_size // 8]
        samples_ddim, _ = sampler.sample(
            S=args.sampling_steps,
            conditioning=c,
            batch_size=args.sample_batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=args.scale,
            unconditional_conditioning=uc,
            eta=0.0,  # deterministic sampling
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = ms.ops.clip_by_value((x_samples_ddim + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
        x_samples_ddim_numpy = x_samples_ddim.asnumpy()

        for x_sample in x_samples_ddim_numpy:
            x_sample = 255.0 * x_sample.transpose(1, 2, 0)
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(class_images_dir, f"{cur_class_images:05}.png"))
            cur_class_images += 1
            if cur_class_images > args.num_class_images:
                break
        if cur_class_images > args.num_class_images:
            break

    end_time = time.time()

    logger.info(
        f"It took {end_time-start_time:.2f} seconds to generate {num_new_images} \
            new images which are saved in: {class_images_dir}."
    )
    del model


def main(args):
    # init
    rank_id, device_id, device_num = init_env(args)
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))
    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        generate_class_images(args)  # inistiate a new model. After image generation, the new model is deleted.
    else:
        logger.info(
            "With with_prior_preservation=False, it turns to vanilla finetuning, and dreambooth is not applied."
        )

    model_config = OmegaConf.load(args.model_config).model
    model_config["params"]["cond_stage_trainable"] = args.train_text_encoder  # overwrites the model_config
    if args.use_lora:
        model_config["params"]["cond_stage_trainable"] = False  # only lora params are trainable
    model_config["params"]["prior_loss_weight"] = args.prior_loss_weight if args.with_prior_preservation else 0.0
    latent_diffusion_with_loss = instantiate_from_config(model_config)
    load_pretrained_model(
        args.pretrained_model_path, latent_diffusion_with_loss, unet_initialize_random=args.unet_initialize_random
    )

    # lora injection
    if args.use_lora:
        # freeze network
        for param in latent_diffusion_with_loss.model.get_parameters():
            param.requires_grad = False

        # inject lora params
        num_injected_params = 0
        injected_params_names = []
        if args.lora_ft_unet:
            unet_lora_layers, unet_lora_params = inject_trainable_lora(
                latent_diffusion_with_loss,
                rank=args.lora_rank,
                use_fp16=args.lora_fp16,
            )
            num_injected_params += len(unet_lora_params)
            injected_params_names.extend(unet_lora_params)
        if args.lora_ft_text_encoder:
            text_encoder_lora_layers, text_encoder_lora_params = inject_trainable_lora_to_textencoder(
                latent_diffusion_with_loss,
                rank=args.lora_rank,
                use_fp16=args.lora_fp16,
            )
            num_injected_params += len(text_encoder_lora_params)
            injected_params_names.extend(text_encoder_lora_params)
        for p in latent_diffusion_with_loss.trainable_params():
            if p.name not in injected_params_names:
                print(f"found {p.name} not lora param but trainable")

        assert (
            len(latent_diffusion_with_loss.trainable_params()) == num_injected_params
        ), "Only lora params {} should be trainable. but got {} trainable params".format(
            num_injected_params, len(latent_diffusion_with_loss.trainable_params())
        )

    # Get tokenizer
    tokenizer = latent_diffusion_with_loss.cond_stage_model.tokenizer

    # build dataloader
    train_dataloader = load_data(
        args.instance_data_dir,
        args.class_data_dir,
        args.instance_prompt,
        args.class_prompt,
        args.train_batch_size,
        tokenizer,
        image_size=args.image_size,
        random_crop=args.random_crop,
        train_data_repeats=args.train_data_repeats,
        rank_id=rank_id,
        with_prior_preservation=args.with_prior_preservation,
    )

    # build optimizer
    optimizer = build_optimizer(
        model=latent_diffusion_with_loss,
        name=args.optim,
        betas=args.betas,
        weight_decay=args.weight_decay,
        lr=args.start_learning_rate,
    )

    loss_scaler = DynamicLossScaleUpdateCell(
        loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
    )

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
        drop_overflow_update=True,  # TODO: allow config
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    model = Model(net_with_grads)

    # callbacks
    callback = [TimeMonitor(args.callback_size), LossMonitor(args.callback_size)]
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)

    if rank_id == 0:
        ckpt_dir = os.path.join(args.output_path, "ckpt", f"rank_{str(rank_id)}")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        save_cb = EvalSaveCallback(
            network=latent_diffusion_with_loss,  # save all
            use_lora=args.use_lora,
            rank_id=rank_id,
            ckpt_save_dir=ckpt_dir,
            ema=ema,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=10,
            ckpt_save_interval=args.ckpt_save_interval,
            lora_rank=args.lora_rank,
            log_interval=args.callback_size,
            record_lr=False,  # LR retrival is not supportted on 910b currently
        )

        callback.append(save_cb)

    # log
    if rank_id == 0:
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"Instance Data path: {args.instance_data_dir}",
                f"Instance Prompt: {args.instance_prompt}",
                f"Class Data path: {args.class_data_dir}",
                f"Class Prompt: {args.class_prompt}",
                f"Precision: {latent_diffusion_with_loss.model.diffusion_model.dtype}",
                f"Use LoRA: {args.use_lora}",
                f"LoRA rank: {args.lora_rank}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.train_batch_size}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"Train Text Encoder: {args.train_text_encoder}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")

    # train
    model.train(
        args.epochs,
        train_dataloader,
        callbacks=callback,
        dataset_sink_mode=args.dataset_sink_mode,
        initial_epoch=start_epoch,
    )


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
