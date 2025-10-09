"""Train step wrapper supporting setting drop overflow update, ema etc"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Literal, Optional

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.mint.distributed import get_world_size
from mindspore.parallel._utils import _get_parallel_mode

from mindone.diffusers.training_utils import GradClipper, GradScaler
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


class TrainStep(nn.Cell, metaclass=ABCMeta):
    """TrainStep with ema and clip grad.

    Args:
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
        gradient_accumulation_steps=1,
        max_grad_norm=None,
        zero_helper=None,
        ema: Optional[EMA] = None,
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.loss_scaler = loss_scaler

        self.ema = ema

        if max_grad_norm is not None:
            self.clip_grad = True
            max_grad_norm = float(max_grad_norm)
            assert max_grad_norm > 0.0 and isinstance(
                max_grad_norm, float
            ), f"clip_norm must be float > 1.0, but got {max_grad_norm}"
        else:
            self.clip_grad = False

        self.max_grad_norm = max_grad_norm

        assert gradient_accumulation_steps >= 1
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.grad_scaler = GradScaler(self.loss_scaler)
        self.grad_clipper = GradClipper(self.max_grad_norm)

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

        self.forward_and_backward = ms.value_and_grad(self.forward, None, weights=self.weights, has_aux=True)
        gradient_checkpointing = kwargs.get("gradient_checkpointing", False)
        if gradient_checkpointing:
            self.recompute(self.network)
            logger.info("Gradient Checkpointing is applied to model.")

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute(parallel_optimizer_comm_recompute=True)
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        elif ms.get_context("mode") == ms.GRAPH_MODE:
            b.add_flags(output_no_recompute=True)

    def get_grad_reducer(self):
        grad_reducer = nn.Identity()
        # if training is distributed
        group_size = get_world_size()
        if group_size != 1:
            grad_reducer = nn.DistributedGradReducer(self.optimizer.parameters)
        return grad_reducer

    def scale_loss(self, loss):
        loss = loss / self.gradient_accumulation_steps
        loss = self.grad_scaler.scale(loss)
        return loss

    def unscale_loss(self, loss):
        return self.grad_scaler.unscale(loss)

    @abstractmethod
    def forward(self, *args, **kwargs):
        # You need to scale the loss when performing the model forward pass to create scaled gradients.
        # Do **NOT** forget to include 'loss = self.scale_loss(loss)' after loss calculation!
        ...

    def train_one_step(self, *inputs):
        return self.construct(*inputs)

    def construct(self, *inputs):
        # 1. compute gradients (of the up-scaled loss w.r.t. the model weights)
        outputs, grads = self.forward_and_backward(*inputs)
        # 1.1 if zero_helper is not None, compute gradients
        if self.zero_helper is not None:
            grads = self.zero_helper.cal_gradients(grads)

        # 2. unscale grads
        grads = self.grad_scaler.unscale(grads)

        # 3. update params if gradients are valid or no dropout=
        # 3.1 gradient accumulation -> clip grads and updates when counter % acc_steps=0 -> reset states
        if self.gradient_accumulation_steps > 1:
            # accumulate grads to inner grads
            self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
            if self.counter % self.gradient_accumulation_steps == 0:
                grads = self.grad_reducer(self.inner_grads)

                grads = self.grad_clipper.clip_grad_norm(grads)

                self.grad_scaler.step(self.run_optimizer, grads)  # will skip update params if gradients have overflow
                # zero_grad
                self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)
                # reset counter to 1
                ops.assign(self.counter, Tensor(1, ms.int32))

            ops.assign_add(self.counter, Tensor(1, ms.int32))
        else:
            # 3.2 clip grads and updates
            grads = self.grad_reducer(grads)
            grads = self.grad_clipper.clip_grad_norm(grads)
            self.grad_scaler.step(self.run_optimizer, grads)  # will skip update params if gradients have overflow

        # 4. Updates the scale for next iteration.
        self.grad_scaler.update()
        # 5. EMA update if not None
        if self.ema is not None:
            self.ema.ema_update()

        # The first item of outputs is loss. Unscales the loss for outside logging.
        loss = self.unscale_loss(outputs[0])
        outputs = (loss,) + outputs[1:]
        return outputs


class TrainStepMmaDA(TrainStep):
    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop("config", None)
        assert self.config is not None, "Must pass configuration via key arguments!"

        super().__init__(*args, **kwargs)

    def forward(
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
        # DO NOT forget to scale loss!
        loss = self.scale_loss(loss)
        return loss, logits, loss_t2i, loss_lm, loss_mmu
