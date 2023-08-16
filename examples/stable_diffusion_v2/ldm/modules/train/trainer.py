"""Train step wrapper supporting setting drop overflow update, ema etc"""
import os

import numpy as np

import mindspore as ms
import mindspore.context as context
from mindspore import Model, Parameter, Tensor, nn, ops
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._ps_context import _cache_enable, _enable_distributed_mindrt, _is_role_pserver, _is_role_sched
from mindspore.parallel._utils import _reset_op_id_with_offset
from mindspore.train.callback import RunContext

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(
        grad.indices,
        grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
        grad.dense_shape,
    )


def _transfer_tensor_to_tuple(inputs):
    """
    If the input is a tensor, convert it to a tuple. If not, the output is unchanged.
    """
    if isinstance(inputs, Tensor):
        return (inputs,)

    return inputs


class TrainOneStepWrapper(nn.TrainOneStepWithLossScaleCell):
    """TrainStep with ema and clip grad.

    Args:
        drop_overflow_update: if True, network will not be updated when gradient is overflow.
        scale_sense (Union[Tensor, Cell]): If this value is a Cell, it will be called
            to update loss scale. If this value is a Tensor, the loss scale can be modified by `set_sense_scale`,
            the shape should be :math:`()` or :math:`(1,)`.

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
        scale_sense=1.0,
        ema=None,
        updates=0,
        drop_overflow_update=True,
        gradient_accumulation_steps=1,
        clip_grad=False,
        clip_norm=1.0,
        verbose=False,
    ):
        super().__init__(network, optimizer, scale_sense)
        self.ema = ema
        self.drop_overflow_update = drop_overflow_update

        assert isinstance(clip_grad, bool), f"Invalid type of clip_grad, got {type(clip_grad)}, expected bool"
        assert clip_norm > 0.0 and isinstance(clip_norm, float), f"clip_norm must be float > 1.0, but got {clip_norm}"
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm

        assert gradient_accumulation_steps >= 1
        self.grad_accu_steps = gradient_accumulation_steps
        if gradient_accumulation_steps > 1:
            # additionally caches network trainable parameters. overhead caused.
            # TODO: try to store it in CPU memory instead of GPU/NPU memory.
            self.accumulated_grads = optimizer.parameters.clone(prefix="grad_accumulated_", init="zeros")
            self.zeros = optimizer.parameters.clone(prefix="zeros_", init="zeros")
            self.cur_accu_step = Parameter(Tensor(0, ms.int32), "grad_accumulate_step_", requires_grad=False)
            self.zero = Tensor(0, ms.int32)
            for p in self.accumulated_grads:
                p.requires_grad = False
            for z in self.zeros:
                z.requires_grad = False

        self.verbose = verbose
        self.is_cpu_device = context.get_context("device_target") == "CPU"  # to support CPU in CI

        self.map = ops.Map()
        self.partial = ops.Partial()

    def construct(self, *inputs):
        # compute loss
        weights = self.weights
        loss = self.network(*inputs)  # mini-batch loss
        scaling_sens = self.scale_sense

        # check loss overflow
        if not self.is_cpu_device:
            status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        else:
            status = None

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))  # loss scale value

        # 1. compute gradients (of the up-scaled loss w.r.t. the model weights)
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)

        # 2. down-scale gradients by loss_scale. grads = grads / scaling_sense  / grad_accu_steps
        # also divide gradients by accumulation steps to avoid taking mean of  the accumulated gradients later
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens * self.grad_accu_steps), grads)

        # 3. check gradient overflow
        if not self.is_cpu_device:
            cond = self.get_overflow_status(status, grads)
            overflow = self.process_loss_scale(cond)
        else:
            overflow = ms.Tensor(False)
            cond = ms.Tensor(False)

        # accumulate gradients and update model weights if no overflow or allow to update even when overflow
        if (not self.drop_overflow_update) or (not overflow):
            # 4. gradient accumulation if enabled
            if self.grad_accu_steps > 1:
                # self.accumulated_grads += grads
                loss = F.depend(loss, self.map(self.partial(ops.assign_add), self.accumulated_grads, grads))
                # self.cur_accu_step += 1
                loss = F.depend(loss, ops.assign_add(self.cur_accu_step, Tensor(1, ms.int32)))

                if self.cur_accu_step % self.grad_accu_steps == 0:
                    # 5. gradient reduction on distributed GPUs/NPUs
                    grads = self.grad_reducer(self.accumulated_grads)

                    # 6. clip grad
                    if self.clip_grad:
                        grads = ops.clip_by_global_norm(grads, self.clip_norm)
                    # 7. optimize
                    loss = F.depend(loss, self.optimizer(grads))

                    # clear gradient accumulation states
                    loss = F.depend(
                        loss, self.map(self.partial(ops.assign), self.accumulated_grads, self.zeros)
                    )  # self.accumulated_grads = 0
                    loss = F.depend(loss, ops.assign(self.cur_accu_step, self.zero))  # self.cur_accu_step = 0
                else:
                    # update LR in each gradient step but not optimize net parameter to ensure the LR curve is
                    # consistent
                    loss = F.depend(loss, self.optimizer.get_lr())  # .get_lr() will make lr step increased by 1
            else:
                # 5. gradient reduction on distributed GPUs/NPUs
                grads = self.grad_reducer(grads)
                # 6. clip grad
                if self.clip_grad:
                    grads = ops.clip_by_global_norm(grads, self.clip_norm)
                # 7. optimize
                loss = F.depend(loss, self.optimizer(grads))

            # 8.ema
            if self.ema is not None:
                self.ema.ema_update()
        else:
            # print("WARNING: Gradient overflow! update skipped.")
            pass

        return loss, cond, scaling_sens


class ModelTrain(Model):
    def __init__(self, network, local_step=0, **kwargs):
        Model.__init__(
            self,
            network,
            loss_fn=None,
            optimizer=None,
            metrics=None,
            eval_network=None,
            eval_indexes=None,
            amp_level="O0",
            boost_level="O0",
            **kwargs,
        )
        self.local_step = local_step

    def _train_process(
        self, epoch, train_dataset, list_callback=None, cb_params=None, initial_epoch=0, valid_infos=None
    ):
        """
        Training process. The data would be passed to network directly.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
            initial_epoch (int): Epoch at which to start train, it used for resuming a previous training run.
                                 Default: 0.
        """
        dataset_helper, _ = self._exec_preprocess(
            is_train=True, dataset=train_dataset, dataset_sink_mode=False, epoch_num=(epoch - initial_epoch)
        )
        cb_params.cur_step_num = 0
        cb_params.dataset_sink_mode = False
        run_context = RunContext(cb_params)
        list_callback.on_train_begin(run_context)
        is_embedding_cache_server = _is_role_pserver() and _cache_enable()

        for i in range(initial_epoch, epoch):
            cb_params.cur_epoch_num = i + 1
            self._current_epoch_num = cb_params.cur_epoch_num
            self._current_step_num = 0

            list_callback.on_train_epoch_begin(run_context)

            for next_element in dataset_helper:
                len_element = len(next_element)
                next_element = _transfer_tensor_to_tuple(next_element)
                if self._loss_fn and len_element != 2:
                    raise ValueError(
                        "When 'loss_fn' is not None, 'train_dataset' should return "
                        "two elements, but got {}, please check the number of elements "
                        "returned by 'train_dataset'".format(len_element)
                    )
                cb_params.cur_step_num += 1
                if i == initial_epoch and cb_params.cur_step_num <= self.local_step:
                    continue

                self._current_step_num = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)

                cb_params.train_dataset_element = next_element
                list_callback.on_train_step_begin(run_context)
                self._check_network_mode(self._train_network, True)
                outputs = self._train_network(*next_element)
                cb_params.net_outputs = outputs
                if self._loss_scale_manager and self._loss_scale_manager.get_drop_overflow_update():
                    overflow = outputs[1]
                    overflow = np.all(overflow.asnumpy())
                    self._loss_scale_manager.update_loss_scale(overflow)

                list_callback.on_train_step_end(run_context)
                if _is_role_sched():
                    os._exit(0)
                # Embedding cache server only run one step.
                if is_embedding_cache_server:
                    break
                should_stop = run_context.get_stop_requested()
                if should_stop:
                    break

            # When it's distributed training and using MindRT,
            # the node id should be reset to start from 0.
            # This is to avoid the timeout when finding the actor route tables in 'train' and 'eval' case(or 'fit').
            if _enable_distributed_mindrt():
                _reset_op_id_with_offset()

            self._eval_during_train(valid_infos, cb_params, list_callback)

            train_dataset.reset()

            # if param is cache enable, flush data from cache to host before epoch end
            self._flush_from_cache(cb_params)

            # Embedding cache server need not do epoch end callback, this process only run one step.
            if not is_embedding_cache_server:
                list_callback.on_train_epoch_end(run_context)
            if "metrics" in cb_params or "eval_results" in cb_params:
                cb_params.pop("metrics", None)
                cb_params.pop("eval_results", None)
            should_stop = run_context.get_stop_requested()
            if should_stop:
                break

        list_callback.on_train_end(run_context)
