import logging
import os
from typing import List, Optional

from mindspore import Parameter, ParameterTuple, RunContext
from mindspore.train import Callback

from mindone.trainers.checkpoint import CheckpointManager

__all__ = ["SaveCkptCallback"]

logger = logging.getLogger("")


class SaveCkptCallback(Callback):
    def __init__(
        self,
        rank_id: Optional[int] = None,
        output_dir: str = "./outputs",
        ckpt_max_keep: int = 5,
        ckpt_save_interval: int = 1,
        save_ema: bool = False,
        ckpt_save_policy: str = "latest_k",
    ) -> None:
        self.rank_id = 0 if rank_id is None else rank_id
        if self.rank_id != 0:
            return

        self.ckpt_save_interval = ckpt_save_interval
        self.save_ema = save_ema

        ckpt_save_dir = os.path.join(output_dir, f"rank_{rank_id}")
        if not os.path.isdir(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)
        self.ckpt_manager = CheckpointManager(ckpt_save_dir, ckpt_save_policy=ckpt_save_policy, k=ckpt_max_keep)

        if self.save_ema:
            self.ema_ckpt_manager = CheckpointManager(ckpt_save_dir, ckpt_save_policy=ckpt_save_policy, k=ckpt_max_keep)

    def on_train_epoch_end(self, run_context: RunContext) -> None:
        if self.rank_id != 0:
            return

        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        epoch_num = cb_params.epoch_num

        if (cur_epoch % self.ckpt_save_interval == 0) or (cur_epoch == epoch_num):
            ckpt_name = f"epoch_{cur_epoch}.ckpt"
            network_weight = cb_params.train_network.network
            self.ckpt_manager.save(network=network_weight, ckpt_name=ckpt_name, perf=cb_params.net_outputs)
            if self.save_ema:
                ckpt_name = f"epoch_{cur_epoch}_ema.ckpt"
                ema_weight = self._drop_ema_prefix(cb_params.train_network.ema.ema_weight)
                self.ema_ckpt_manager.save(network=ema_weight, ckpt_name=ckpt_name, perf=cb_params.net_outputs)

    def _drop_ema_prefix(self, weight: ParameterTuple) -> List[Parameter]:
        new_weight = list()
        for x in weight:
            x.name = x.name.replace("ema.", "")
            new_weight.append(x)
        return new_weight
