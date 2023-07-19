# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import time

import mindspore as ms
from mindspore.train.callback._callback import Callback

from .checkpoint import CheckpointManager


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
        ckpt_save_interval=1,
        lora_rank=None,
    ):
        self.rank_id = rank_id
        self.is_main_device = rank_id in [0, None]
        self.ema = ema
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_interval = ckpt_save_interval
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)

        self.last_epoch_end_time = time.time()
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()

        if self.is_main_device:
            self.ckpt_save_policy = ckpt_save_policy
            self.ckpt_manager = CheckpointManager(
                ckpt_save_dir,
                ckpt_save_policy,
                k=ckpt_max_keep,
            )

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

    def on_train_epoch_begin(self, run_context):
        """
        Called before each epoch beginning.
        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()

    def on_train_epoch_end(self, run_context):
        """
        Called after each training epoch end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        epoch_num = cb_params.epoch_num

        # data_sink_mode = cb_params.dataset_sink_mode
        # if data_sink_mode:
        #     loss_scale_manager = cb_params.train_network.network.loss_scaling_manager
        # else:
        #     loss_scale_manager = cb_params.train_network.loss_scaling_manager

        if self.is_main_device:
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
                """
                ms.save_checkpoint(
                    cb_params.train_network,
                    os.path.join(self.ckpt_save_dir, "train_resume.ckpt"),
                    append_dict={"epoch_num": cur_epoch, "loss_scale": loss_scale_manager.get_loss_scale()},
                )
                """

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
