"""Train step wrapper supporting setting drop overflow update, ema etc"""
import logging
from typing import Literal, Optional

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.amp import all_finite
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.mint.distributed import get_world_size
from mindspore.parallel._utils import _get_parallel_mode

from mindone.trainers.ema import EMA
from mindone.trainers.zero import ZeroHelper, prepare_network

logger = logging.getLogger(__name__)


def do_ckpt_combine_online(net_to_save, optimizer_parallel_group=None):
    new_net_to_save = []
    all_gather_op = ops.AllGather(optimizer_parallel_group)
    if optimizer_parallel_group is None:
        logger.warning("Not set zero group, set it WORLD_COMM_GROUP.")
        optimizer_parallel_group = GlobalComm.WORLD_COMM_GROUP
    for item in net_to_save:
        param = item["data"]
        if param.parallel_optimizer:
            new_data = ms.Tensor(all_gather_op(param).asnumpy())
        else:
            new_data = ms.Tensor(param.asnumpy())
        new_net_to_save.append({"name": param.name, "data": new_data})
    return new_net_to_save


def prepare_train_network(
    network: nn.Cell,
    optimizer: nn.Optimizer,
    zero_stage: Literal[0, 1, 2, 3] = 0,
    optimizer_offload: bool = False,
    optimizer_parallel_group: str = None,
    dp_group: str = None,
    comm_fusion: dict = None,
    parallel_modules=None,
):
    """
    Prepare network and optimizer for distributed training.

    Args:
        network (`nn.Cell`): train network, not include grad function,
            grad function must be built after rewrite train network.
        optimizer (`nn.Optimizer`): Must be the subclass of MindSpore Optimizer.
        zero_stage (`int`, *optional*): Stage setting of ZeRO, default is 0.
        optimizer_offload (`bool`, *optional*): Only take effect when optimizer is AdamWeightDecay, default is False.
        optimizer_parallel_group (`str`, *optional*): The name of the optimizer parallel communication group, default is None.
        dp_group (`str`, *optional*): The name of the data parallel communication group, default is None.
        comm_fusion (`dict`, *optional*): A dict contains the types and configurations
            for setting the communication fusion, default is None, turn off the communication fusion. If set a dict,
            turn on the communication fusion.
            Examples: {"allreduce": {"openstate": True, "bucket_size": 5e8},
                       "reduce_scatter": {"openstate": True, "bucket_size": 5e8},
                       "allgather": {"openstate": False, "bucket_size": 5e8},}
        parallel_modules (`dict`, *optional*): A dict of Cells could split parameters in zero3, default is None.
            If None, use `PARALLEL_MODULES` from `mindone.models.modules.parallel`.
    """
    if zero_stage not in [0, 1, 2, 3]:
        raise ValueError("Not support zero_stage {zero_stage}")
    if optimizer_parallel_group is None:
        logger.warning("Not set zero group, set it WORLD_COMM_GROUP.")
        optimizer_parallel_group = GlobalComm.WORLD_COMM_GROUP
    if optimizer_parallel_group != GlobalComm.WORLD_COMM_GROUP and dp_group is None:
        raise ValueError(
            "optimizer_parallel_group {optimizer_parallel_group} and dp_group {dp_group} not full network hccl group coverage"
        )

    is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
    if not is_parallel and zero_stage == 0:
        logger.info("No need prepare train_network with zero.")
        zero_helper = None
    else:
        network = prepare_network(network, zero_stage, optimizer_parallel_group, parallel_modules=parallel_modules)
        zero_helper = ZeroHelper(
            optimizer, zero_stage, optimizer_parallel_group, dp_group, optimizer_offload, comm_fusion
        )
    return network, zero_helper


class TrainOneStepWrapper:
    """TrainStep with ema and clip grad.

    Args:
        drop_overflow_update: if True, network will not be updated when gradient is overflow.
        scale_sense (Union[Tensor, Cell]): If this value is a Cell, it will be called
            to update loss scale. If this value is a Tensor, the loss scale can be modified by `set_sense_scale`,
            the shape should be :math:`()` or :math:`(1,)`.
        zero_helper (class): Zero redundancy optimizer(ZeRO) build helper, default is None.

    Returns:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scale value.
        loss (Tensor) -  A scalar, the loss value.
        overflow (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        loss scale (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.

    """

    def __init__(
        self,
        network,
        optimizer,
        loss_scaler=None,
        ema: Optional[EMA] = None,
        drop_overflow_update=True,
        gradient_accumulation_steps=1,
        clip_grad=False,
        clip_norm=1.0,
        zero_helper=None,
        config=None,
    ):
        self.network = network
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.loss_scaler = loss_scaler
        self.config = config
        assert self.config is not None, "Expect to have configuration but got None!"

        self.ema = ema
        self.drop_overflow_update = drop_overflow_update
        clip_norm = float(clip_norm)

        assert isinstance(clip_grad, bool), f"Invalid type of clip_grad, got {type(clip_grad)}, expected bool"
        assert clip_norm > 0.0 and isinstance(clip_norm, float), f"clip_norm must be float > 1.0, but got {clip_norm}"
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm

        assert gradient_accumulation_steps >= 1
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # zero init
        self.zero_helper = zero_helper
        self.zero_stage = zero_helper.zero_stage if zero_helper is not None else 0
        self.run_optimizer = zero_helper.run_optimizer if zero_helper is not None else self.optimizer
        self.grad_reducer = self.get_grad_reducer() if self.zero_stage == 0 else nn.Identity()
        if self.zero_stage != 0:
            self.zero_helper.split_params()

        # gradient accumulator
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init="zeros")
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init="zeros")
        self.counter = ms.Parameter(Tensor(1, ms.int32), "counter_")
        self.map = ops.HyperMap()

        self.value_and_grad = ms.value_and_grad(self.forward_fn, None, weights=self.weights, has_aux=True)
        if hasattr(self.config.model, "gradient_checkpointing") and self.config.model.gradient_checkpointing:
            self.recompute(self.network)
            logger.info(f"Gradient Checkpointing during training: {config.model.gradient_checkpointing}")

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute(parallel_optimizer_comm_recompute=True)
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        elif ms.get_context("mode") == ms.GRAPH_MODE:
            b.add_flags(output_no_recompute=True)

    def forward_fn(
        self,
        input_ids: ms.Tensor,
        labels: ms.Tensor,
        batch_size_t2i: int,
        batch_size_lm: int,
        batch_size_mmu: int,
        p_mask_lm: ms.Tensor,
        p_mask_mmu: ms.Tensor,
        answer_lengths: ms.Tensor,
        t2i_masks: ms.Tensor,
    ):
        logits, loss_t2i, loss_lm, loss_mmu = self.network.construct_process(
            input_ids=input_ids,
            labels=labels,
            batch_size_t2i=batch_size_t2i,
            batch_size_lm=batch_size_lm,
            batch_size_mmu=batch_size_mmu,
            max_seq_length=self.config.dataset.preprocessing.max_seq_length,
            p_mask_lm=p_mask_lm,
            p_mask_mmu=p_mask_mmu,
            answer_lengths=answer_lengths,
            t2i_masks=t2i_masks,
        )

        loss = (
            self.config.training.t2i_coeff * loss_t2i
            + self.config.training.lm_coeff * loss_lm
            + self.config.training.mmu_coeff * loss_mmu
        )
        if self.loss_scaler:
            loss = self.loss_scaler.scale(loss)
        loss = loss / self.gradient_accumulation_steps
        return loss, logits, loss_t2i, loss_lm, loss_mmu

    def get_grad_reducer(self):
        grad_reducer = nn.Identity()
        # if training is distributed
        group_size = get_world_size()
        if group_size != 1:
            grad_reducer = nn.DistributedGradReducer(self.optimizer.parameters)
        return grad_reducer

    def set_train(self, mode: bool = True):
        # Delegate the setting of training mode behavior to the network.
        self.network.set_train(mode)

    def train_one_step(self, *inputs):
        # 1. compute gradients (of the up-scaled loss w.r.t. the model weights)
        (loss, logits, loss_t2i, loss_lm, loss_mmu), grads = self.value_and_grad(*inputs)

        # 1.1 if zero_helper is not None, compute gradients
        if self.zero_helper is not None:
            grads = self.zero_helper.cal_gradients(grads)

        grads_finite = True  # assume valid gradients (no overflow)

        # 2. unscale loss and grads, check overflow status
        if self.loss_scaler:
            loss = self.loss_scaler.unscale(loss)
            grads = self.loss_scaler.unscale(grads)

            grads_finite = all_finite(grads)
            self.loss_scaler.adjust(grads_finite)
        # 3. update params if gradients are valid or no dropout
        if grads_finite or (not self.drop_overflow_update):
            # 3.1 gradient accumulation -> clip grads and updates when counter % acc_steps=0 -> reset states
            if self.gradient_accumulation_steps > 1:
                # accumulate grads to inner grads
                self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
                if self.counter % self.gradient_accumulation_steps == 0:
                    grads = self.grad_reducer(self.inner_grads)
                    if self.clip_grad:
                        grads = ops.clip_by_global_norm(grads, self.clip_norm)
                    self.run_optimizer(grads)
                    # reset inner grads to zeros
                    self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)
                    # reset counter to 1
                    ops.assign(self.counter, Tensor(1, ms.int32))

                ops.assign_add(self.counter, Tensor(1, ms.int32))
            else:
                # 3.2 clip grads and updates
                grads = self.grad_reducer(grads)
                if self.clip_grad:
                    grads = ops.clip_by_global_norm(grads, self.clip_norm)
                self.run_optimizer(grads)
        else:
            logger.warning("WARNING: Gradient overflow! update skipped.")

        return loss, logits, loss_t2i, loss_lm, loss_mmu
