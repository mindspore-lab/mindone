import logging

import mindspore as ms
from mindspore import RunContext, Tensor, ops
from mindspore.train.callback._callback import Callback

_logger = logging.getLogger("")  # get the root _logger


class LossMonitor(Callback):
    def __init__(self, log_interval: int = 1, log_overflow: bool = True) -> None:
        self.log_interval = log_interval
        self.log_overflow = log_overflow
        self.step_num = 0

    def on_train_step_begin(self, run_context: RunContext) -> None:
        self.step_num += 1

    def on_train_epoch_end(self, run_context: RunContext) -> None:
        self.step_num = 0

    def on_train_step_end(self, run_context: RunContext) -> None:
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num

        if cur_step % self.log_interval == 0:
            cur_lr = self._fetch_optimizer_lr(cb_params)
            cur_loss = self._fetch_loss(cb_params)
            cur_loss_scale = self._fetch_loss_scale(cb_params)

            _logger.info(
                "epoch: %d step: %d, lr: %.7f, loss: %.6f, loss scale: %d.",
                cb_params.cur_epoch_num,
                self.step_num,
                cur_lr.item(),
                cur_loss.item(),
                cur_loss_scale.item(),
            )

            if self.log_overflow:
                overflow = cb_params.net_outputs[1]
                if overflow:
                    _logger.warning(f"overflow detected in epoch {cb_params.cur_epoch_num} step {self.step_num}.")

    def _get_optimizer_from_cbp(self, cb_params):
        if cb_params.optimizer is not None:
            optimizer = cb_params.optimizer
        elif cb_params.dataset_sink_mode:
            optimizer = cb_params.train_network.network.optimizer
        else:
            optimizer = cb_params.train_network.optimizer
        return optimizer

    def _fetch_loss_scale(self, cb_params) -> Tensor:
        if cb_params.dataset_sink_mode:
            return cb_params.train_network.network.scale_sense
        else:
            return cb_params.train_network.scale_sense

    def _fetch_optimizer_lr(self, cb_params) -> Tensor:
        opt = self._get_optimizer_from_cbp(cb_params)
        lr = opt.learning_rate
        if opt.dynamic_lr:
            lr = opt.learning_rate(ops.clip(opt.global_step - 1, min=0))[0]
        return lr

    def _fetch_loss(self, cb_params) -> Tensor:
        loss = cb_params.net_outputs[0]
        return loss


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
