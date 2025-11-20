import math
import os
from typing import Optional

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from omnigen2.dataset import OmniGen2Collator, OmniGen2TrainDataset
from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2 import OmniGen2TrainPipeline
from omnigen2.transport import create_transport
from omnigen2.utils.callbacks import BinLossCallback, VisualizationCallback
from omnigen2.utils.ema import EMA
from omnigen2.utils.logging_utils import get_logger, setup_logging
from omnigen2.utils.training_utils import init_env, log_time_distribution, prepare_train_network
from transformers import AutoTokenizer

import mindspore as ms
from mindspore.nn import no_init_parameters
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint

from mindone.data import create_dataloader
from mindone.diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from mindone.peft import LoraConfig
from mindone.trainers import create_optimizer, create_scheduler
from mindone.trainers.callback import OverflowMonitor, StopAtStepCallback
from mindone.transformers import Qwen2_5_VLModel as TextEncoder

logger = get_logger(__name__)


def parse_args() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(description="OmniGen2 training script")
    parser.add_argument("-c", "--config", action=ActionConfigFile, help="Path to configuration file (YAML format)")
    parser.add_argument("--name", type=str, default="", help="Name of the experiment")
    parser.add_function_arguments(init_env, "env")
    parser.add_method_arguments(OmniGen2Transformer2DModel, "from_pretrained", "models.transformer")
    parser.add_method_arguments(AutoencoderKL, "from_pretrained", "models.vae")
    parser.add_method_arguments(
        TextEncoder, "from_pretrained", "models.text_encoder", skip={"return_unused_kwargs", "config_file_name"}
    )
    parser.add_function_arguments(create_scheduler, "train.lr_scheduler", skip={"steps_per_epoch", "num_epochs"})
    parser.add_function_arguments(create_optimizer, "train.optimizer", skip={"params", "lr"})
    parser.add_function_arguments(create_transport, "transport", skip={"seq_len", "rank", "world_size"})
    parser.add_class_arguments(OmniGen2TrainDataset, "data", skip={"tokenizer"}, instantiate=False)
    parser.add_function_arguments(
        create_dataloader, "dataloader", skip={"dataset", "transforms", "batch_transforms", "device_num", "rank_id"}
    )
    parser.link_arguments("env.debug", "dataloader.debug", apply_on="parse")
    parser.add_argument("--train.lora", type=Optional[LoraConfig], default=None)
    parser.add_function_arguments(prepare_train_network, "train.settings", skip={"network", "optimizer", "ema"})
    parser.add_subclass_arguments(EMA, "train.ema", instantiate=False, required=False, skip={"network"})
    parser.add_argument("--train.resume_from_checkpoint", type=str)
    parser.add_argument("--train.gradient_checkpointing", type=bool, default=False)
    parser.add_argument("--train.steps", type=int, required=True)
    parser.link_arguments("train.steps", "train.lr_scheduler.total_steps", apply_on="parse")
    parser.add_argument("--collator.maximum_text_tokens", type=int, default=512)
    parser.add_argument("--save.checkpointing_steps", type=int, default=500)
    parser.add_argument("--save.checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--save.train_visualization_steps", type=int, default=100)

    conf = parser.parse_args()
    return conf, parser


def main(args, parser):
    rank, world_size = init_env(**args.env)

    output_dir = os.path.join("experiments", args.name)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        parser.save(args, output_dir + "/config.yaml", format="yaml", overwrite=True)
    setup_logging(logger, output_dir, is_main_process=rank == 0)

    with no_init_parameters():
        # BUG: `mindspore_dtype` has to be `ms.Type` rather than a string
        dtypes = {"float32": ms.float32, "float16": ms.float16, "bfloat16": ms.bfloat16}

        model_args = args.models.transformer.as_dict()
        mindspore_dtype = dtypes[model_args.pop("mindspore_dtype")]
        model = OmniGen2Transformer2DModel.from_pretrained(
            **model_args, subfolder="transformer", mindspore_dtype=mindspore_dtype
        ).set_train(True)

        vae_args = args.models.vae.as_dict()
        mindspore_dtype = dtypes[vae_args.pop("mindspore_dtype")]
        vae = AutoencoderKL.from_pretrained(**vae_args, subfolder="vae", mindspore_dtype=mindspore_dtype).set_train(
            False
        )

        text_encoder_args = args.models.text_encoder.as_dict()
        mindspore_dtype = dtypes[text_encoder_args.pop("mindspore_dtype")]
        text_encoder = TextEncoder.from_pretrained(**text_encoder_args, mindspore_dtype=mindspore_dtype).set_train(
            False
        )
    freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(model.config.axes_dim_rope, model.config.axes_lens, theta=10000)

    ema = EMA(model, **args.train.ema.init_args) if args.train.ema else None

    text_tokenizer = AutoTokenizer.from_pretrained(args.models.text_encoder.pretrained_model_name_or_path)
    text_tokenizer.padding_side = "right"

    if rank == 0:
        text_tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    if rank == 0:
        # BUG: TypeError: Object of type BFloat is not JSON serializable
        text_encoder.config.vision_config.mindspore_dtype = "bfloat16"
        text_encoder.config.text_config.mindspore_dtype = "bfloat16"
        text_encoder.save_pretrained(os.path.join(output_dir, "text_encoder"))

    if args.train.lora is not None:
        for param in model.get_parameters():
            param.requires_grad = False
        # now we will add new LoRA weights the transformer layers
        model.add_adapter(parser.instantiate_classes(args).train.lora)

    if args.train.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # scheduler and optimizer creation
    lr = create_scheduler(steps_per_epoch=0, **args.train.lr_scheduler)
    optimizer = create_optimizer(params=model.trainable_params(), lr=lr, **args.train.optimizer)

    logger.info("***** Prepare dataset *****")
    train_dataset = OmniGen2TrainDataset(tokenizer=text_tokenizer, **args.data)

    # default: 1000 steps, linear noise schedule
    transport = create_transport(
        seq_len=args.data.max_output_pixels // 16 // 16, rank=rank, world_size=world_size, **args.transport
    )
    # Log time distribution for analysis
    if rank == 0:
        log_time_distribution(transport, args, output_dir, rank, world_size)

    logger.info(f"Number of training samples: {len(train_dataset)}")

    logger.info("***** Prepare dataLoader *****")
    train_dataloader = create_dataloader(
        train_dataset,
        batch_transforms={
            "operations": OmniGen2Collator(tokenizer=text_tokenizer, max_token_len=args.collator.maximum_text_tokens),
            "input_columns": ["instruction"],
            "output_columns": ["text_ids", "text_mask"],
        },
        device_num=world_size,
        rank_id=rank,
        **args.dataloader,
    )

    logger.info(f"{args.dataloader.batch_size=} {args.train.settings.gradient_accumulation_steps=} {world_size=}")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.settings.gradient_accumulation_steps)

    # Train!
    total_batch_size = args.dataloader.batch_size * world_size * args.train.settings.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {args.dataloader.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.train.settings.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.train.steps}")
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.train.resume_from_checkpoint:
        if args.train.resume_from_checkpoint != "latest":
            path = os.path.basename(args.train.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            print(f"Checkpoint '{args.train.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.train.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            print(f"Resuming from checkpoint {path}")
            param_dict = ms.load_checkpoint(os.path.join(output_dir, str(path)))
            # TODO: verify
            param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict)
            if param_not_load or ckpt_not_load:
                logger.warning(
                    f"Exist ckpt params not loaded: {ckpt_not_load} (total: {len(ckpt_not_load)}),\n"
                    f"or net params not loaded: {param_not_load} (total: {len(param_not_load)})"
                )
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    pipeline = OmniGen2TrainPipeline(text_encoder, vae, model, freqs_cis, transport)
    pipeline = prepare_train_network(pipeline, optimizer=optimizer, ema=ema, **args.train.settings)

    callbacks: list[Callback] = [OverflowMonitor(), BinLossCallback(batch_size=args.dataloader.batch_size)]
    if rank == 0:
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=args.save.checkpointing_steps,
            keep_checkpoint_max=args.save.checkpoints_total_limit,
            saved_network=model,
            append_info=["step_num"],
            exception_save=True,
        )
        callbacks.append(ModelCheckpoint(directory=output_dir, config=ckpt_config))
        callbacks.append(
            VisualizationCallback(output_dir, vae, text_tokenizer, frequency=args.save.train_visualization_steps)
        )

    callbacks.append(StopAtStepCallback(train_steps=args.train.steps, global_step=initial_global_step))

    # 6. train
    logger.info("Start training...")
    model = ms.Model(pipeline)
    # train() uses epochs, so the training will be terminated by the StopAtStepCallback
    model.train(args.train.steps, train_dataloader, callbacks=callbacks, initial_epoch=first_epoch)


if __name__ == "__main__":
    args, parser = parse_args()
    main(args, parser)
