import logging
import os
import time
from typing import List, Literal, Optional, Tuple, Union

from mindspore import Profiler, Tensor, nn, save_checkpoint
from mindspore.communication import get_rank
from mindspore.train.callback._callback import Callback, _handle_loss

from .checkpoint import CheckpointManager
from .ema import EMA
from .recorder import PerfRecorder

_logger = logging.getLogger("")

__all__ = ["OverflowMonitor", "EvalSaveCallback", "ProfilerCallback", "StopAtStepCallback"]


def get_real_rank():
    """get rank id"""
    try:
        return get_rank()
    except RuntimeError:
        return int(os.getenv("RANK_ID", "0"))


class OverflowMonitor(Callback):
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
        network: nn.Cell,
        use_lora: bool = False,
        rank_id: int = 0,
        ckpt_save_dir: str = "./",
        output_dir: str = None,
        ema: EMA = None,
        save_ema_only: bool = True,
        ckpt_save_policy: Literal["top_k", "latest_k", None] = "latest_k",
        monitor_metric: Optional[str] = None,
        ckpt_max_keep: int = 10,
        step_mode: bool = False,
        ckpt_save_interval: int = 1,
        use_step_unit: bool = False,
        data_sink_mode: bool = True,
        lora_rank: Optional[int] = None,
        log_interval: int = 1,
        start_epoch: int = 0,
        record_lr: bool = True,
        model_name: str = "sd",
        save_trainable_only: bool = False,
        param_save_filter: List[str] = None,
        resume_prefix_blacklist: Optional[Union[str, Tuple[str, ...]]] = None,
        integrated_save: bool = False,
        save_training_resume: bool = True,
        train_steps: int = -1,
        prefer_low_perf: bool = False,
    ):
        """
        Args:
            step_mode: if True, ckpt_save_interval is counted in steps. otherwise, in epochs.
            param_save_filter: indicates what parameters to save in checkpoint. If None, save all parameters in network. \
                Otherwise, only params that contain one of the keyword in param_save_filter list will be saved.
            resume_prefix_blacklist: exclude parameters with one of these prefixes to be saved in resume checkpoint,
                                     e.g. ('swap.', 'vae.').
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
        self.save_ema_only = save_ema_only

        if self.is_main_device:
            self.ckpt_save_policy = ckpt_save_policy
            self.monitor_metric = monitor_metric
            self.ckpt_manager = CheckpointManager(
                ckpt_save_dir,
                ckpt_save_policy,
                k=ckpt_max_keep,
                integrated_save=integrated_save,
                prefer_low_perf=prefer_low_perf,
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

        self.use_step_unit = use_step_unit
        self.train_steps = train_steps
        self.save_training_resume = save_training_resume
        self.choice_func = None
        if resume_prefix_blacklist:
            if isinstance(resume_prefix_blacklist, str):
                resume_prefix_blacklist = (resume_prefix_blacklist,)
            self.choice_func = lambda x: not x.startswith(resume_prefix_blacklist)

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = _handle_loss(cb_params.net_outputs)
        opt = self._get_optimizer_from_cbp(cb_params)
        cur_step = int(opt.global_step.asnumpy().item())
        if cur_step <= 0:
            cur_step = cb_params.cur_step_num + self.start_epoch * cb_params.batch_num

        step_num = (cb_params.batch_num * cb_params.epoch_num) if self.train_steps < 0 else self.train_steps

        if cur_step % cb_params.batch_num == 0:
            cur_epoch = cb_params.cur_epoch_num
        else:
            cur_epoch = cb_params.cur_epoch_num - 1

        if self.is_main_device:
            # if data sink, train step callback will not be invokded
            if self.step_mode and (cur_step % self.ckpt_save_interval == 0 or cur_step == step_num):
                ckpt_name = (
                    f"{self.model_name}-s{cur_step}.ckpt"
                    if self.use_step_unit
                    else f"{self.model_name}-e{cur_epoch}.ckpt"
                )

                append_dict = {"lora_rank": self.lora_rank} if self.use_lora else None
                perf = cb_params.get("eval_results")
                if perf or self.ckpt_save_policy != "top_k":
                    if perf:
                        perf = perf[self.monitor_metric]
                    if self.ema is not None:
                        if not self.save_ema_only:
                            self.ckpt_manager.save(
                                self.net_to_save,
                                perf,
                                ckpt_name=ckpt_name.replace(".ckpt", "_nonema.ckpt"),
                                append_dict=append_dict,
                            )
                        # swap ema weight and network weight
                        self.ema.swap_before_eval()

                    # save history checkpoints
                    self.ckpt_manager.save(self.net_to_save, perf, ckpt_name=ckpt_name, append_dict=append_dict)

                if self.save_training_resume:
                    # TODO: resume training for step.
                    save_checkpoint(
                        cb_params.train_network,
                        os.path.join(self.ckpt_save_dir, "train_resume.ckpt"),
                        choice_func=self.choice_func,
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

                if self.record_lr:
                    _logger.info(
                        "epoch %d, step %d, lr %.7f, loss %.6f, loss scale %d, global_step %d, step_time(ms) %.1f",
                        cb_params.cur_epoch_num,
                        (cb_params.cur_step_num - 1) % cb_params.batch_num + 1,
                        cur_lr.asnumpy().item(),
                        loss.asnumpy().item(),
                        self._get_scaling_value_from_cbp(cb_params),
                        cur_step,
                        (train_time * 1000) / self.log_interval,
                    )
                else:
                    _logger.info(
                        "epoch %d, step %d, loss %.6f, loss scale %d, global_step %d, step_time(ms) %.1f",
                        cb_params.cur_epoch_num,
                        (cb_params.cur_step_num - 1) % cb_params.batch_num + 1,
                        loss.asnumpy().item(),
                        self._get_scaling_value_from_cbp(cb_params),
                        cur_step,
                        (train_time * 1000) / self.log_interval,
                    )

                self.step_start_time = time.time()

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

        # cur_step = cur_epoch * cb_params.batch_num
        opt = self._get_optimizer_from_cbp(cb_params)
        cur_step = int(opt.global_step.asnumpy().item())

        if self.is_main_device and (not self.step_mode):
            if (cur_epoch % self.ckpt_save_interval == 0) or (cur_epoch == epoch_num):
                ckpt_name = (
                    f"{self.model_name}-s{cur_step}.ckpt"
                    if self.use_step_unit
                    else f"{self.model_name}-e{cur_epoch}.ckpt"
                )

                append_dict = {"lora_rank": self.lora_rank} if self.use_lora else None
                if self.ema is not None:
                    if not self.save_ema_only:
                        self.ckpt_manager.save(
                            self.net_to_save,
                            None,
                            ckpt_name=ckpt_name.replace(".ckpt", "_nonema.ckpt"),
                            append_dict=append_dict,
                        )
                    # swap ema weight and network weight
                    self.ema.swap_before_eval()

                # save history checkpoints
                self.ckpt_manager.save(
                    self.net_to_save, perf=cb_params["net_outputs"], ckpt_name=ckpt_name, append_dict=append_dict
                )

                if self.save_training_resume:
                    save_checkpoint(
                        cb_params.train_network,
                        os.path.join(self.ckpt_save_dir, "train_resume.ckpt"),
                        choice_func=self.choice_func,
                        append_dict={
                            "epoch_num": cur_epoch,
                            "loss_scale": self._get_scaling_value_from_cbp(cb_params),
                        },
                    )

                # swap back network weight and ema weight. MUST execute after model saving and before next-step training
                if self.ema is not None:
                    self.ema.swap_after_eval()

            self.last_epoch_end_time = time.time()

    def on_train_end(self, run_context):
        if self.is_main_device:
            if self.ckpt_save_policy == "top_k":
                log_str = f"Top K checkpoints: \n{self.main_indicator}\tcheckpoint\n"
                for p, ckpt_name in self.ckpt_manager.get_ckpt_queue():
                    log_str += f"{p: .4f}\t{os.path.join(self.ckpt_save_dir, ckpt_name)}\n"

    def on_eval_end(self, run_context):
        if self.is_main_device:
            cb_params = run_context.original_args()
            metrics = cb_params.get("metrics")
            if metrics is not None:
                metrics = {k: f"{v: .4f}" for k, v in metrics.items()}
                _logger.info(f"Eval result epoch {cb_params.cur_epoch_num}: {metrics}")

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

    def _fetch_optimizer_lr(self, cb_params) -> Tensor:
        opt = self._get_optimizer_from_cbp(cb_params)
        lr = opt.learning_rate
        if opt.dynamic_lr:
            lr = opt.learning_rate(opt.global_step - 1)[0]
        return lr


class StopAtStepCallback(Callback):
    # stop the training process when reach train_steps
    def __init__(self, train_steps, global_step=0):
        self.global_step = global_step
        self.train_steps = train_steps

    def on_train_step_end(self, run_context):
        self.global_step += 1
        if self.global_step >= self.train_steps:
            run_context.request_stop()


class ProfilerCallback(Callback):
    def __init__(self, start_step=1, end_step=2, exit_after_analyze=True, out_dir="./profiler_data"):
        self.start_step = start_step
        self.end_step = end_step
        self.exit_after_analyze = exit_after_analyze
        rank_id = get_real_rank()
        out_dir = os.path.join(out_dir, f"rank_{rank_id}")
        # If value of profile_framework is not None, a subdirectory named host_info will be generated under the
        # specified profiler directory to store the collected memory and time files on the Host side.
        self.profiler = Profiler(
            start_profile=False, output_path=out_dir, profile_framework="all", data_simplication=False
        )

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


class ProfilerCallbackEpoch(Callback):
    def __init__(self, start_epoch, stop_epoch, output_dir="./profiler_data"):
        super().__init__()
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.profiler = Profiler(start_profile=False, output_path=output_dir)

    def on_train_epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num == self.start_epoch:
            self.profiler.start()

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num == self.stop_epoch:
            self.profiler.stop()
            self.profiler.analyse()
