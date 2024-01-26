import logging
import os
import time
from typing import List

import mindspore as ms
from mindspore.train.callback._callback import Callback, _handle_loss

from .checkpoint import CheckpointManager
from .recorder import PerfRecorder

_logger = logging.getLogger(__name__)


class OverflowMonitor(ms.Callback):
    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        overflow = cb_params.net_outputs[1]
        if overflow:
            _logger.warning(f"overflow detected in epoch {cur_epoch_num} step {cur_step_in_epoch}")
        return super().step_end(run_context)


class EvalSaveCallback(Callback):
    def __init__(
        self,
        network,
        use_lora=False,
        rank_id=0,
        ckpt_save_dir="./",
        output_dir=None,
        ema=None,
        ckpt_save_policy="lastest_k",
        ckpt_max_keep=10,
        step_mode=False,
        ckpt_save_interval=1,
        lora_rank=None,
        log_interval=10,
        start_epoch=0,
        record_lr=True,
        model_name="sd",
        save_trainable_only: bool = False,
        param_save_filter: List[str] = None,
    ):
        """
        Args:
            param_save_filter: indicates what parameters to save in checkpoint. If None, save all parameters in network. \
                Otherwise, only params that contain one of the keyword in param_save_filter list will be saved.
        """
        self.rank_id = rank_id
        self.is_main_device = rank_id in [0, None]
        self.ema = ema
        if output_dir is not None:
            self.output_dir = output_dir
            self.ckpt_save_dir = os.path.join(output_dir, "ckpt")
        else:
            self.output_dir = ckpt_save_dir.replace("/ckpt", "")
            self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_interval = ckpt_save_interval
        self.step_mode = step_mode
        self.model_name = model_name
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)

        self.last_epoch_end_time = time.time()
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()
        self.log_interval = log_interval
        self.start_epoch = start_epoch
        self.record_lr = record_lr

        if self.is_main_device:
            self.ckpt_save_policy = ckpt_save_policy
            self.ckpt_manager = CheckpointManager(
                ckpt_save_dir,
                ckpt_save_policy,
                k=ckpt_max_keep,
            )
            if self.start_epoch == 0:
                if self.record_lr:
                    perf_columns = ["step", "loss", "lr", "train_time(s)"]
                else:
                    perf_columns = ["step", "loss", "train_time(s)"]
                self.rec = PerfRecorder(self.output_dir, metric_names=perf_columns)
            else:
                self.rec = PerfRecorder(self.output_dir, resume=True)

        self.save_trainable_only = save_trainable_only or use_lora
        if self.save_trainable_only:
            # save lora trainable params only
            self.net_to_save = [{"name": p.name, "data": p} for p in network.trainable_params()]
            self.lora_rank = lora_rank
        elif param_save_filter is not None:
            if isinstance(param_save_filter, str):
                param_save_filter = [param_save_filter]
            self.net_to_save = []
            for p in network.get_parameters():
                for keyword in param_save_filter:
                    if keyword in p.name:
                        self.net_to_save.append({"name": p.name, "data": p})
                        break
        else:
            self.net_to_save = network
        self.use_lora = use_lora

    """
    def on_train_begin(self, run_context):
        # TOOD: remove it after debug
        cb_params = run_context.original_args()
        epoch_num  = cb_params.epoch_num

        if self.is_main_device:
                if self.ema is not None:
                    # swap ema weight and network weight
                    self.ema.swap_before_eval()

                # save history checkpoints
                self.ckpt_manager.save(self.net_to_save, None, ckpt_name=f"{self.model_name}-test.ckpt")
                #ms.save_checkpoint(
                #    cb_params.train_network,
                #    os.path.join(self.ckpt_save_dir, "train_resume.ckpt"),
                #    append_dict={"epoch_num": cur_epoch, "loss_scale": loss_scale_manager.get_loss_scale()},
                #)

                # swap back network weight and ema weight. MUST execute after model saving and before next-step training
                if (self.ema is not None) and eval_done:
                    self.ema.swap_after_eval()
    """

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = _handle_loss(cb_params.net_outputs)
        cur_step = cb_params.cur_step_num + self.start_epoch * cb_params.batch_num
        step_num = cb_params.batch_num * cb_params.epoch_num
        if cur_step % cb_params.batch_num == 0:
            cur_epoch = cb_params.cur_epoch_num
        else:
            cur_epoch = cb_params.cur_epoch_num - 1

        if self.is_main_device:
            if self.step_mode and (cur_step % self.ckpt_save_interval == 0 or cur_step == step_num):
                if self.ema is not None:
                    # swap ema weight and network weight
                    self.ema.swap_before_eval()
                    # print('DEBUG: Store ema weights to save checkpoint.')

                # save history checkpoints
                append_dict = {"lora_rank": self.lora_rank} if self.use_lora else None
                self.ckpt_manager.save(
                    self.net_to_save, None, ckpt_name=f"{self.model_name}-{cur_step}.ckpt", append_dict=append_dict
                )

                # TODO: resume training for step.
                ms.save_checkpoint(
                    cb_params.train_network,
                    os.path.join(self.ckpt_save_dir, "train_resume.ckpt"),
                    append_dict={
                        "epoch_num": cur_epoch,
                        "cur_step": cur_step,
                        "loss_scale": self._get_scaling_value_from_cbp(cb_params),
                    },
                )

                # swap back network weight and ema weight. MUST execute after model saving and before next-step training
                if self.ema is not None:
                    self.ema.swap_after_eval()

            if cur_step % self.log_interval == 0 or cur_step == step_num:
                if self.record_lr:
                    cur_lr = self._fetch_optimizer_lr(cb_params)  # get lr

                train_time = time.time() - self.step_start_time
                step_pref_value = (
                    [cur_step, loss, cur_lr, train_time] if self.record_lr else [cur_step, loss, train_time]
                )
                self.rec.add(*step_pref_value)

                self.step_start_time = time.time()
                if self.record_lr:
                    _logger.info(
                        "epoch: %d step: %d, lr: %.7f, loss: %.6f, loss scale: %d, average step time: %.6f.",
                        cb_params.cur_epoch_num,
                        (cb_params.cur_step_num - 1) % cb_params.batch_num + 1,
                        cur_lr.asnumpy().item(),
                        loss.asnumpy().item(),
                        self._get_scaling_value_from_cbp(cb_params),
                        train_time / self.log_interval,
                    )
                else:
                    _logger.info(
                        "epoch: %d step: %d, loss: %.6f, loss scale: %d, average step time: %.6f.",
                        cb_params.cur_epoch_num,
                        (cb_params.cur_step_num - 1) % cb_params.batch_num + 1,
                        loss.asnumpy().item(),
                        self._get_scaling_value_from_cbp(cb_params),
                        train_time / self.log_interval,
                    )

    def on_train_epoch_begin(self, run_context):
        """
        Called before each epoch beginning.
        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, run_context):
        """
        Called after each training epoch end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        epoch_num = cb_params.epoch_num

        if self.is_main_device and not self.step_mode:
            if (cur_epoch % self.ckpt_save_interval == 0) or (cur_epoch == epoch_num):
                if self.ema is not None:
                    # swap ema weight and network weight
                    self.ema.swap_before_eval()
                    # print('DEBUG: Store ema weights to save checkpoint.')

                # save history checkpoints
                append_dict = {"lora_rank": self.lora_rank} if self.use_lora else None
                self.ckpt_manager.save(
                    self.net_to_save, None, ckpt_name=f"{self.model_name}-{cur_epoch}.ckpt", append_dict=append_dict
                )

                ms.save_checkpoint(
                    cb_params.train_network,
                    os.path.join(self.ckpt_save_dir, "train_resume.ckpt"),
                    append_dict={
                        "epoch_num": cur_epoch,
                        "loss_scale": self._get_scaling_value_from_cbp(cb_params),
                    },
                )

                # swap back network weight and ema weight. MUST execute after model saving and before next-step training
                if self.ema is not None:
                    self.ema.swap_after_eval()

            # tot_time = time.time() - self.last_epoch_end_time
            self.last_epoch_end_time = time.time()

    def on_train_end(self, run_context):
        if self.is_main_device:
            if self.ckpt_save_policy == "top_k":
                log_str = f"Top K checkpoints:\n{self.main_indicator}\tcheckpoint\n"
                for p, ckpt_name in self.ckpt_manager.get_ckpt_queue():
                    log_str += f"{p:.4f}\t{os.path.join(self.ckpt_save_dir, ckpt_name)}\n"

    def _get_optimizer_from_cbp(self, cb_params):
        if cb_params.optimizer is not None:
            optimizer = cb_params.optimizer
        elif cb_params.dataset_sink_mode:
            optimizer = cb_params.train_network.network.optimizer
        else:
            optimizer = cb_params.train_network.optimizer
        return optimizer

    def _get_scaling_value_from_cbp(self, cb_params):
        if cb_params.dataset_sink_mode:
            return cb_params.train_network.network.scale_sense.asnumpy().item()
        else:
            return cb_params.train_network.scale_sense.asnumpy().item()

    def _fetch_optimizer_lr(self, cb_params):
        opt = self._get_optimizer_from_cbp(cb_params)
        lr = opt.learning_rate
        if opt.dynamic_lr:
            lr = opt.learning_rate(opt.global_step - 1)[0]
            # lr = opt.learning_rate.asnumpy()(int(opt.global_step.asnumpy()) - 1)[0]
        return lr


class ProfilerCallback(ms.Callback):
    def __init__(self, start_step=1, end_step=2, exit_after_analyze=True, out_dir="./profiler_data"):
        self.start_step = start_step
        self.end_step = end_step
        self.exit_after_analyze = exit_after_analyze
        self.profiler = ms.Profiler(start_profile=False, output_path=out_dir)

    def on_train_step_begin(self, run_context):
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if cur_step == self.start_step:
            _logger.info(f"start analyzing profiler in step range [{self.start_step}, {self.end_step}]")
            self.profiler.start()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if cur_step == self.end_step:
            self.profiler.stop()
            self.profiler.analyse()
            _logger.info(f"finish analyzing profiler in step range [{self.start_step}, {self.end_step}]")
            if self.exit_after_analyze:
                run_context.request_stop()
