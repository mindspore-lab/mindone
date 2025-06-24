"""Train step wrapper supporting setting drop overflow update, ema etc"""
import logging
from typing import Optional

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.amp import all_finite
from mindspore.mint.distributed import get_world_size

from mindone.trainers.ema import EMA

logger = logging.getLogger(__name__)


# @ms.jit_class
class Accumulator:
    def __init__(self, optimizer, accumulate_step):
        self.optimizer = optimizer

        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init="zeros")
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init="zeros")
        self.counter = ms.Parameter(Tensor(1, ms.int32), "counter_")
        assert accumulate_step > 0
        self.accumulate_step = accumulate_step
        self.map = ops.HyperMap()

    def __call__(self, grads):
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        if self.counter % self.accumulate_step == 0:
            self.optimizer(self.inner_grads)
            self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)

        ops.assign_add(self.counter, Tensor(1, ms.int32))

        return True


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
        self.accumulator = Accumulator(self.optimizer, gradient_accumulation_steps)

        # zero init
        self.zero_helper = zero_helper
        self.zero_stage = zero_helper.zero_stage if zero_helper is not None else 0
        self.run_optimizer = zero_helper.run_optimizer if zero_helper is not None else self.optimizer
        self.grad_reducer = self.get_grad_reducer() if self.zero_stage == 0 else nn.Identity()
        if self.zero_stage != 0:
            self.zero_helper.split_params()
            if gradient_accumulation_steps > 1:
                self.accumulated_grads = optimizer.parameters.clone(prefix="grad_accumulated_", init="zeros")

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
        grads_finite = True  # valid gradients (no overflow)
        if self.loss_scaler:
            loss = self.loss_scaler.unscale(loss)
            grads = self.loss_scaler.unscale(grads)

            grads_finite = all_finite(grads)
            self.loss_scaler.adjust(grads_finite)

        grads = self.grad_reducer(grads)
        if self.clip_grad:
            grads = ops.clip_by_global_norm(grads, self.clip_norm)
        if grads_finite or (not self.drop_overflow_update):
            self.accumulator(grads)
        else:
            logger.warning("WARNING: Gradient overflow! update skipped.")

        return loss, logits, loss_t2i, loss_lm, loss_mmu
