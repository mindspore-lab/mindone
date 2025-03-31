import logging
import os
import time

from models import Embed1, Embed2, build_vae_var
from utils.arg_utils import parse_args
from utils.data import creat_dataset
from utils.lr_control import lr_wd_annealing
from utils.net_with_loss import GeneratorWithLoss
from utils.optim import create_optimizer
from utils.utils import load_from_checkpoint

import mindspore as ms
from mindspore import nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

from mindone.trainers.checkpoint import CheckpointManager, resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.recorder import PerfRecorder
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.env import init_train_env
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def create_loss_scaler(loss_scaler_type, init_loss_scale, loss_scale_factor=2, scale_window=1000):
    if loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=init_loss_scale, scale_factor=loss_scale_factor, scale_window=scale_window
        )
    elif loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(init_loss_scale)
    else:
        raise ValueError

    return loss_scaler


def main(args):
    # init
    device_id, rank_id, device_num = init_train_env(
        args.ms_mode,
        seed=args.seed,
        distributed=args.use_parallel,
        jit_level=args.jit_level,
        max_device_memory=args.max_device_memory,
    )
    set_random_seed(args.seed)
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # dataset
    ld_train = creat_dataset(
        args.data_path,
        final_reso=args.data_load_reso,
        batch_size=args.batch_size,
        is_training=True,
        hflip=args.hflip,
        mid_reso=args.mid_reso,
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=args.num_parallel_workers,
        drop_remainder=True,
    )

    dataset_size = ld_train.get_dataset_size()

    # vae and var
    vae_local, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        patch_nums=args.patch_nums,
        num_classes=args.num_classes,
        depth=args.depth,
        shared_aln=args.saln,
        attn_l2_norm=args.anorm,
        init_adaln=args.aln,
        init_adaln_gamma=args.alng,
        init_head=args.hd,
        init_std=args.ini,
    )
    if args.vae_checkpoint:
        load_from_checkpoint(vae_local, args.vae_checkpoint)
    else:
        logger.warning("VAE uses random initialization!")

    if args.var_checkpoint:
        load_from_checkpoint(var, args.var_checkpoint)
    else:
        logger.warning("VAR uses random initialization!")

    dtype_map = {"fp16": ms.float16, "bf16": ms.bfloat16}
    if args.dtype in ["fp16", "bf16"]:
        var = auto_mixed_precision(var, amp_level="O2", dtype=dtype_map[args.dtype], custom_fp32_cells=[Embed1, Embed2])

    # build net with loss
    var_with_loss = GeneratorWithLoss(
        patch_nums=args.patch_nums,
        vae_local=vae_local,
        var=var,
        label_smooth=args.ls,
    )

    # build lr
    max_step = args.epochs * dataset_size
    wp_it = args.wp * dataset_size
    args.tlr = args.gradient_accumulation_steps * args.tblr * args.batch_size * device_num / 256
    lr = lr_wd_annealing(args.scheduler, args.tlr, wp_it, max_step, wp0=args.wp0, wpe=args.wpe)

    # build optimizer
    optim_var = create_optimizer(
        var_with_loss.trainable_params(),
        name=args.optim,
        betas=args.betas,
        nowd_keys={
            "cls_token",
            "start_token",
            "task_token",
            "cfg_uncond",
            "pos_embed",
            "pos_1LC",
            "pos_start",
            "start_pos",
            "lvl_embed",
            "gamma",
            "beta",
            "ada_gss",
            "moe_bias",
            "scale_mul",
        },
        weight_decay=args.weight_decay,
        lr=lr,
    )

    loss_scaler_var = create_loss_scaler(
        args.loss_scaler_type, args.init_loss_scale, args.loss_scale_factor, args.scale_window
    )

    ema = (
        EMA(
            var,
            ema_decay=args.ema_decay,
            offloading=False,
        )
        if args.use_ema
        else None
    )

    # resume training states
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch, cur_iter = 0, 0
    if args.resume:
        resume_ckpt = os.path.join(ckpt_dir, "train_resume.ckpt") if isinstance(args.resume, bool) else args.resume

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
            var_with_loss, optim_var, resume_ckpt
        )
        loss_scaler_var.loss_scale_value = loss_scale
        loss_scaler_var.cur_iter = cur_iter
        loss_scaler_var.last_overflow_iter = last_overflow_iter
        logger.info(f"Resume training from {resume_ckpt}")

    training_step_var = TrainOneStepWrapper(
        var_with_loss,
        optimizer=optim_var,
        scale_sense=loss_scaler_var,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    if args.ckpt_save_steps > 0:
        save_by_step = True
    else:
        save_by_step = False

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

    ds_iter = ld_train.create_tuple_iterator(num_epochs=args.epochs - start_epoch)

    prog_wp_it = args.pgwp * dataset_size

    if rank_id == 0:
        tot_params, trainable_params = count_params(var_with_loss)
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"Mindspore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
                f"Distributed mode: {args.use_parallel}",
                f"Model depth: {args.depth}",
                f"Total params: {tot_params}, Traninable params: {trainable_params}",
                f"VAR dtype: {args.dtype}",
                f"Data path: {args.data_path}",
                f"Batch size: {args.batch_size}",
                f"Num batches: {dataset_size}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)
    global_step = start_epoch * dataset_size
    global_step = ms.Tensor(global_step, dtype=ms.int32)
    for epoch in range(start_epoch, args.epochs):
        start_time_e = time.time()

        for step, data in enumerate(ds_iter):
            start_time_s = time.time()
            inp = data[0]
            label = data[1]

            wp_it = args.wp * dataset_size
            if args.pg:
                if global_step <= wp_it:
                    prog_si = args.pg0
                elif global_step > max_step * args.pg:
                    prog_si = len(args.patch_nums) - 1
                else:
                    delta = len(args.patch_nums) - 1 - args.pg0
                    progress = min(max((global_step - wp_it) / (max_step * args.pg - wp_it), 0), 1)
                    prog_si = args.pg0 + round(progress * delta)
            else:
                prog_si = -1

            loss, overflow, scaling_sens = training_step_var(inp, label, prog_si, prog_wp_it)
            global_step = global_step + 1

            if overflow:
                logger.warning(f"Overflow occurs in step {global_step}")

            # log
            step_time = time.time() - start_time_s
            if step % args.log_interval == 0 or step == dataset_size:
                loss = float(loss.asnumpy())
                logger.info(
                    f"E: {epoch + 1}, S: {step + 1}, Loss: {loss:.4f}, Global step {global_step}, "
                    f"Step time: {step_time * 1000:.2f}ms"
                )

            if rank_id == 0:
                step_pref_value = [global_step, loss, step_time]
                record.add(*step_pref_value)

        epoch_cost = time.time() - start_time_e
        per_step_time = epoch_cost / dataset_size
        cur_epoch = epoch + 1
        logger.info(
            f"Epoch:[{int(cur_epoch):>3d}/{int(args.epochs):>3d}], "
            f"epoch time:{epoch_cost:.2f}s, per step time:{per_step_time * 1000:.2f}ms, "
        )

        if save_by_step and rank_id == 0:
            if (global_step % args.ckpt_save_steps == 0) or (cur_epoch == args.epochs):
                ckpt_name = f"var-d{args.depth}-s{global_step}.ckpt"
                if ema is not None:
                    ema.swap_before_eval()

                ckpt_manager.save(var, None, ckpt_name=ckpt_name, append_dict=None)
                if ema is not None:
                    ema.swap_after_eval()

        if not save_by_step and rank_id == 0:
            if (cur_epoch % args.ckpt_save_interval == 0) or (cur_epoch == args.epochs):
                ckpt_name = f"var-d{args.depth}-e{cur_epoch}.ckpt"
                if ema is not None:
                    ema.swap_before_eval()

                ckpt_manager.save(var, None, ckpt_name=ckpt_name, append_dict=None)
                if ema is not None:
                    ema.swap_after_eval()


if __name__ == "__main__":
    args = parse_args()

    main(args)
