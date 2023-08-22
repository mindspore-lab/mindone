import os
import time

import mindspore as ms
from mindspore.train.callback._callback import Callback, _handle_loss

from .checkpoint import CheckpointManager
from .recorder import PerfRecorder


class OverflowMonitor(ms.Callback):
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        overflow = cb_params.net_outputs[1]
        if overflow:
            print(f"overflow detected in epoch {cur_epoch_num} step {cur_step_in_epoch}")
        return super().step_end(run_context)


class EvalSaveCallback(Callback):
    def __init__(
        self,
        network,
        use_lora=False,
        rank_id=0,
        ckpt_save_dir="./",
        ema=None,
        ckpt_save_policy="lastest_k",
        ckpt_max_keep=10,
        step_mode=False,
        ckpt_save_interval=1,
        lora_rank=None,
        log_interval=10,
        start_epoch=0,
        start_step=0,
    ):
        self.rank_id = rank_id
        self.is_main_device = rank_id in [0, None]
        self.ema = ema
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_interval = ckpt_save_interval
        self.step_mode = step_mode
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)

        self.last_epoch_end_time = time.time()
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()
        self.log_interval = log_interval
        self.start_epoch = start_epoch
        self.start_step = start_step

        if self.is_main_device:
            self.ckpt_save_policy = ckpt_save_policy
            self.ckpt_manager = CheckpointManager(
                ckpt_save_dir,
                ckpt_save_policy,
                k=ckpt_max_keep,
            )
            if self.start_epoch == 0 and self.start_step == 0:
                perf_columns = ["step", "loss", "train_time(s)"]
                self.rec = PerfRecorder(self.ckpt_save_dir, metric_names=perf_columns)
            else:
                self.rec = PerfRecorder(self.ckpt_save_dir, resume=True)

        if use_lora:
            # save lora trainable params only
            self.net_to_save = [{"name": p.name, "data": p} for p in network.trainable_params()]
            self.lora_rank = lora_rank
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
                self.ckpt_manager.save(self.net_to_save, None, ckpt_name=f"sd-test.ckpt")
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
        step_num = cb_params.batch_num * (cb_params.epoch_num + self.start_epoch)
        if cur_step % cb_params.batch_num == 0:
            cur_epoch = cb_params.cur_epoch_num
        else:
            cur_epoch = cb_params.cur_epoch_num - 1

        data_sink_mode = cb_params.dataset_sink_mode
        if data_sink_mode:
            loss_scale_manager = cb_params.train_network.network.loss_scaling_manager
        else:
            loss_scale_manager = cb_params.train_network.loss_scaling_manager

        if self.is_main_device:
            if self.step_mode and (cur_step % self.ckpt_save_interval == 0 or cur_step == step_num):
                if self.ema is not None:
                    # swap ema weight and network weight
                    self.ema.swap_before_eval()
                    # print('DEBUG: Store ema weights to save checkpoint.')

                # save history checkpoints
                append_dict = {"lora_rank": self.lora_rank} if self.use_lora else None
                self.ckpt_manager.save(self.net_to_save, None, ckpt_name=f"sd-{cur_step}.ckpt", append_dict=append_dict)

                ms.save_checkpoint(
                    cb_params.train_network,
                    os.path.join(self.ckpt_save_dir, "train_resume.ckpt"),
                    append_dict={
                        "epoch_num": cur_epoch,
                        "cur_step": cur_step,
                        "loss_scale": loss_scale_manager.get_loss_scale(),
                    },
                )

                # swap back network weight and ema weight. MUST execute after model saving and before next-step training
                if self.ema is not None:
                    self.ema.swap_after_eval()

            if cur_step % self.log_interval == 0 or cur_step == step_num:
                train_time = time.time() - self.step_start_time
                step_pref_value = [cur_step, loss, train_time]
                self.rec.add(*step_pref_value)

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

        data_sink_mode = cb_params.dataset_sink_mode
        if data_sink_mode:
            loss_scale_manager = cb_params.train_network.network.loss_scaling_manager
        else:
            loss_scale_manager = cb_params.train_network.loss_scaling_manager

        if self.is_main_device and not self.step_mode:
            if (cur_epoch % self.ckpt_save_interval == 0) or (cur_epoch == epoch_num):
                if self.ema is not None:
                    # swap ema weight and network weight
                    self.ema.swap_before_eval()
                    # print('DEBUG: Store ema weights to save checkpoint.')

                # save history checkpoints
                append_dict = {"lora_rank": self.lora_rank} if self.use_lora else None
                self.ckpt_manager.save(
                    self.net_to_save, None, ckpt_name=f"sd-{cur_epoch}.ckpt", append_dict=append_dict
                )

                ms.save_checkpoint(
                    cb_params.train_network,
                    os.path.join(self.ckpt_save_dir, "train_resume.ckpt"),
                    append_dict={"epoch_num": cur_epoch, "loss_scale": loss_scale_manager.get_loss_scale()},
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
