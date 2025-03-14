"""
Reproduce JanusPro training
"""
import argparse

# import datetime
import logging
import math
import os
import sys
import time

import yaml

import mindspore as ms
from mindspore import nn
from mindspore._c_expression import reset_op_id
from mindspore.communication.management import get_group_size, get_rank, init

# from mindspore.nn.utils import no_init_parameters

# from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
# from mindspore.train.callback import TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
# sys.path.insert(0, ".")

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.models.modeling_vlm import MultiModalityConfig
from janus.train.lr_schedule import WarmupCosineDecayLR
from janus.train.t2i_dataset import create_dataloader_t2i
from janus.train.text_dataset import create_dataloader_text
from janus.train.vqa_dataset import create_dataloader_vqa
from janus.utils.io import set_model_param_dtype

from mindone.trainers.checkpoint import CheckpointManager

# from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.recorder import PerfRecorder
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.transformers.mindspore_adapter.clip_grad import clip_grad_norm

# from mindone.trainers.zero import prepare_train_network
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(mode, seed, distribute=False):
    set_random_seed(seed)
    # ms.set_context(max_device_memory=max_device_memory)
    ms.set_context(mode=mode)
    ms.set_context(jit_config={"jit_level": "O0"})

    if distribute:
        ms.set_context(mode=mode)
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
        ms.reset_auto_parallel_context()

        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        device_num = 1
        rank_id = 0
        ms.set_context(mode=mode)

    return rank_id, device_num


def main(args):
    # 0. env init
    # time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # args.output_path = os.path.join(args.output_path, time_str)

    rank_id, device_num = init_env(
        args.ms_mode,
        args.seed,
        distribute=args.use_parallel,
    )

    set_logger(name="", output_dir=args.output_path, rank=rank_id)

    # 1. janus model init
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)

    config = MultiModalityConfig.from_pretrained(args.model_path)
    config.torch_dtype = args.dtype
    config.language_config.torch_dtype = args.dtype
    config.language_config._attn_implementation = "flash_attention_2"  # use FA by default
    if args.load_weight:
        vl_gpt = MultiModalityCausalLM.from_pretrained(args.model_path, config=config)
    else:
        # with no_init_parameters():
        vl_gpt = MultiModalityCausalLM(config=config)

    if args.ckpt_path is not None:
        parameter_dict = ms.load_checkpoint(args.ckpt_path)
        param_not_load, ckpt_not_load = ms.load_param_into_net(vl_gpt, parameter_dict, strict_load=True)
        logger.info("net param not load: {}".format(param_not_load))
        logger.info("ckpt param not load: {}".format(ckpt_not_load))

    # 1.1 mixed precision
    dtype_map = {"float16": ms.float16, "bfloat16": ms.bfloat16}
    dtype = dtype_map[args.dtype]
    if args.dtype != "float32":
        vl_gpt = set_model_param_dtype(vl_gpt, dtype)

    # 1.2 set trainable parameters (refer to Janus paper)
    # TODO: use config.yaml to set traning strategy
    num_frozen_params = 0
    num_train_params = 0
    all_modules = set(
        [
            vl_gpt.vision_model,
            vl_gpt.gen_vision_model,
            vl_gpt.language_model,
            vl_gpt.aligner,
            vl_gpt.gen_aligner,
            vl_gpt.gen_head,
            vl_gpt.gen_embed,
        ]
    )
    if args.training_stage == 1:
        # Stage I: Training adaptors and image head
        # freeze sigLIP, VQ16, llm; train adaptors and image head
        frozen_modules = set([vl_gpt.vision_model, vl_gpt.gen_vision_model, vl_gpt.language_model])
    elif args.training_stage == 2:
        # Stage II: unfied pretraining
        # further unfreeze llm
        frozen_modules = set([vl_gpt.vision_model, vl_gpt.gen_vision_model])
    elif args.training_stage == 3:
        # Stage III: SFT
        # only freeze gen. vision autoencoder(VQ); train all others: gen adaptor, und. enc (sliLIP) + und. adaptor, LLM, text head, image head
        # TODO: whether gen_embed (nn.Embed) should be trainable in stage 3
        frozen_modules = set([vl_gpt.gen_vision_model])
    else:
        raise NotImplementedError

    trainable_modules = all_modules - frozen_modules

    for module in frozen_modules:
        module.set_train(False)
        for param in module.get_parameters():
            param.requires_grad = False
            num_frozen_params += 1

    for module in trainable_modules:
        module.set_train(True)
        for param in module.get_parameters():
            param.requires_grad = True
            num_train_params += 1

    # VQ encoder doesn't need grad
    vl_gpt.gen_vision_model.set_grad(requires_grad=False)

    # debug to check gradient influence: set token embedding table for text and image to be non-trainable
    freeze_embed_tables = args.freeze_embedding
    if freeze_embed_tables:
        for module in (vl_gpt.gen_embed, vl_gpt.language_model.model.embed_tokens):
            module.set_train(False)
            for param in module.get_parameters():
                param.requires_grad = False
    tot_params = len(list(vl_gpt.get_parameters()))
    print(f"tot params: {tot_params}, trainable params: {num_train_params}, frozen params: {num_frozen_params}")
    assert num_frozen_params + num_train_params == tot_params, "All params should be set to trainable or frozen."
    # 1.3 save the model config
    config.save_pretrained(args.output_path)

    # 2. prepare dataset and loader
    # FIXME: output task_type in dataloader
    task = args.task
    if task == "text":
        # FIXME: allow setting path
        dataloader = create_dataloader_text(
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            vl_chat_processor=vl_chat_processor,
            max_token_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_samples=args.num_samples,
        )
    elif task == "vqa":
        dataloader = create_dataloader_vqa(
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            vl_chat_processor=vl_chat_processor,
            max_token_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_samples=args.num_samples,
        )
    elif task == "t2i":
        dataloader = create_dataloader_t2i(
            csv_path=args.csv_path,
            data_dir=args.data_dir,
            vl_chat_processor=vl_chat_processor,
            max_token_length=args.max_length,
            image_size=args.image_size,
            null_prompt_prob=args.null_prompt_prob,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_samples=args.num_samples,
        )
    else:
        raise NotImplementedError
    # task_map = {"text": 0, "vqa": 1, "t2i": 2}

    # 3. setup trainer and config hyper-params
    # loss_scaler = nn.FixedLossScaleUpdateCell(1024)  # tune
    optimizer = ms.mint.optim.AdamW(
        vl_gpt.trainable_params(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-6,
    )
    assert args.warmup_steps < args.train_steps
    scheduler = WarmupCosineDecayLR(
        optimizer,
        lr_max=args.learning_rate,
        lr_min=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.train_steps - args.warmup_steps,
    )

    use_value_and_grad = args.use_value_and_grad
    if use_value_and_grad:

        def forward_fn(data):
            loss = vl_gpt(*data)
            return loss

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
        if args.use_parallel:
            grad_reducer = nn.DistributedGradReducer(optimizer.parameters)

        def train_step(data):
            loss, grads = grad_fn(data)
            if args.use_parallel:
                grads = grad_reducer(grads)

            # FIXME: after adding this branch, time cost increase by ~150ms/step
            if args.clip_grad:
                grads = clip_grad_norm(grads, args.max_grad_norm)

            optimizer(grads)

            return loss

    else:
        train_step = TrainOneStepWrapper(
            vl_gpt,
            optimizer=optimizer,
            scale_sense=ms.Tensor(1.0),  # tune
            clip_grad=True,  # tune
            clip_norm=5.0,  # tune
            # ema=ema,
            # zero_stage=args.zero_stage,
        )

    # TODO: for sequence parallel, save ckpt for other ranks
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    # TODO: suppor training resume
    start_epoch = 0
    start_global_step = 0
    if rank_id == 0:
        ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        perf_columns = ["step", "loss", "train_time(s)"]
        output_dir = ckpt_dir.replace("/ckpt", "")
        if start_epoch == 0:
            record = PerfRecorder(output_dir, metric_names=perf_columns)
        else:
            record = PerfRecorder(output_dir, resume=True)

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

        logger.info("Start training...")

    # 4. training loop
    start_time_s = time.time()

    # ds_iter = dataloader.create_tuple_iterator(num_epochs=num_epochs - start_epoch)
    ds_iter = dataloader.create_tuple_iterator(num_epochs=-1)
    num_batches = dataloader.get_dataset_size()
    num_epochs = math.ceil(args.train_steps / num_batches)
    global_step = start_global_step

    for epoch in range(start_epoch + 1, num_epochs + 1):
        # for step in range(args.train_steps):
        for step, data in enumerate(ds_iter, 1):
            """
            data = (ms.Tensor(input_ids, dtype=ms.int32),
                ms.Tensor(labels, dtype=ms.int32),
                ms.Tensor(attention_masks, dtype=ms.bool_),
                ms.Tensor(image_seq_masks, dtype=ms.bool_),
                ms.Tensor(image, dtype=dtype),
                )
            """
            data[-1] = data[-1].to(dtype)  # image pixel values cast to bfloat16

            if use_value_and_grad:
                loss = train_step(data)
            else:
                loss, overflow, scaling_sens = train_step(*data)

            step_time = time.time() - start_time_s
            global_step += 1
            loss_val = float(loss.asnumpy())

            scheduler.step()
            cur_lr = scheduler.get_last_lr()[0].asnumpy()
            # print("lr", [lr for lr in optimizer.lrs])

            logger.info(
                f"epoch {epoch}, step {step}, loss {loss_val:.8f}, lr {cur_lr:.7f}, step time {step_time*1000:.2f}ms"
            )

            if rank_id == 0:
                step_pref_value = [global_step, loss_val, step_time]
                record.add(*step_pref_value)

            if (global_step > 0) and (global_step % args.ckpt_save_steps == 0):
                ckpt_name = f"model-s{global_step}.ckpt"
                ckpt_manager.save(vl_gpt, None, ckpt_name=ckpt_name, append_dict=None)
            start_time_s = time.time()

            # TODO: allow stop at the last step

    logger.info(f"Finished training. check results in {args.output_path}")
    reset_op_id()
    logger.info("End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_mode", type=int, default=1, help="mindspore mode, 0: graph, 1: pynative")
    # TODO: support model_name "deepseek-ai/Janus-Pro-1B" for simplicity
    parser.add_argument("--model_path", type=str, default="ckpts/Janus-Pro-1B", help="path to Janus model")
    parser.add_argument(
        "--training_stage", type=int, default=3, choices=[1, 2, 3], help="model training stage, can be 1, 2, or 3"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to model checkpoint in .ckpt format, if None, will use the pretrained weight in mode_path",
    )
    parser.add_argument(
        "--load_weight", type=str2bool, default=True, help="if True, will not load pretrained weight in model_path"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="model dtype"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--use_value_and_grad",
        default=True,
        type=str2bool,
        help="if False, use mindone wrapped trainer. if True, use custom step based on `value_and_grad` api",
    )
    parser.add_argument(
        "--freeze_embedding",
        default=False,
        type=str2bool,
        help="if Ture, freeze llm embedding table and gen embedding table (nn.Embedding)",
    )
    parser.add_argument(
        "--output_path", default="outputs/janus-sft", type=str, help="output directory to save training results"
    )

    # training hyperparms
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--end_learning_rate", default=1e-5, type=float, help="end learning rate for cosine decay")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="weight decay")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="clip graident")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="max gradient l2 norm")
    parser.add_argument(
        "--null_prompt_prob",
        default=0.0,
        type=float,
        help="probability of replace text caption with empty str for condition-free guidance training in t2i task",
    )
    parser.add_argument("--train_steps", default=5000, type=int, help="training steps")
    parser.add_argument("--warmup_steps", default=50, type=int, help="lr warmup steps")
    parser.add_argument("--ckpt_save_steps", default=500, type=int, help="save ckpt every this step")
    parser.add_argument("--ckpt_max_keep", default=3, type=int, help="num of checkpoints to keep during training")
    parser.add_argument(
        "--max_length",
        default=1024,
        type=int,
        help="sequence max length, input sequence will be padded (left pad) and truncated to this max length",
    )

    # training data config
    parser.add_argument("--task", default="t2i", type=str, help="text, t2i, vqa, or mixed")
    parser.add_argument(
        "--csv_path",
        default="",
        type=str,
        help="path to csv annotation, contain `image_path` and `text_en` column for image path and caption respectively",
    )
    parser.add_argument(
        "--dataset_name", default="", type=str, help="dataset name, used for the right vqa and text dataset loader"
    )
    parser.add_argument(
        "--data_dir",
        default="datasets/",
        type=str,
        help="dataset directory contatining the images specified by `image_path` in csv_path",
    )
    parser.add_argument(
        "--num_samples",
        default=-1,
        type=int,
        help="if -1, train on the whole dataset; if not -1, will pick this number of samples for training.",
    )
    parser.add_argument(
        "--image_size",
        default=384,
        type=int,
        help="image resize and crop to to size. Be cautious to change as Janus is trained using a fix image size of 384",
    )
    parser.add_argument("--shuffle", default=True, type=str2bool, help="shuffle dataset or not")

    args = parser.parse_args()
    main(args)
