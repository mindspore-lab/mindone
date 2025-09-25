import logging

import numpy as np

import mindspore as ms
from mindspore import Parameter, ParameterTuple, Tensor
from mindspore import _checkparam as validator
from mindspore import ops
from mindspore.ops import MultitypeFuncGraph

from .adamw_mint import AdamW, _optim_adamw_opt

_logger = logging.getLogger(__name__)


update_ = MultitypeFuncGraph("update_")


@update_.register("Tensor", "Tensor")
def update_param(source_param: Tensor, target_param: Tensor) -> None:
    target_param.copy_(source_param.to(target_param.dtype))


class BF16AdamW(AdamW):
    r"""Implements the BF16 Adam Weight Decay algorithm.
    This is an AdamW optimizer with a float32 copy of the parameters, in accordance with section C.2 of the
    [Scaling Language Models](https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf) .

    There are several projects that use this optimizer, including
        - ``examples/huanyuanvideo``
        - ``examples/moviegen``
        - ``examples/diffusers/cogvideox_factory``
    You can either use this optimizer directly or by passing ``create_optimizer(..., name="adamw_bf16")``.
    """

    def __init__(
        self,
        params,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-6,
        weight_decay=0.0,
        amsgrad=False,
        *,
        maximize=False,
    ):
        super().__init__(
            params,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )

        # maintain a copy of parameters in fp32 for the optimiser update
        self._master_parameters = []
        for param in self._parameters:
            self._master_parameters.append(
                Parameter(
                    Tensor.from_numpy(param.numpy().astype(np.float32)),
                    name="master." + param.name,
                )
            )
        self._master_parameters = ParameterTuple(self._master_parameters)

    def _check_param_value(self, beta1, beta2, eps, prim_name):
        """Check the type of inputs."""
        validator.check_value_type("beta1", beta1, [float], prim_name)
        validator.check_value_type("beta2", beta2, [float], prim_name)
        validator.check_value_type("eps", eps, [float], prim_name)
        validator.check_float_range(beta1, 0.0, 1.0, validator.INC_NEITHER, "beta1", prim_name)
        validator.check_float_range(beta2, 0.0, 1.0, validator.INC_NEITHER, "beta2", prim_name)
        validator.check_positive_float(eps, "eps", prim_name)
        for x in self.parameters:
            if x.dtype == ms.float32:
                _logger.warning(f"model parameter {x.name} should be `bfloat16`, but got `{x.dtype}`.")

    @ms.jit
    def construct(self, gradients):
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.global_step.add_(self.global_step_increase_tensor)
        state_step = self.global_step.astype(ms.float32)
        if self.amsgrad:
            if self.is_group:
                if self.is_group_lr:
                    optim_result = self.hyper_map(
                        ops.partial(
                            _optim_adamw_opt,
                            self.adamw_opt,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            state_step,
                            self.amsgrad,
                            self.maximize,
                        ),
                        lr,
                        weight_decay,
                        self._master_parameters,
                        gradients,
                        self.exp_avg,
                        self.exp_avg_sq,
                        self.max_exp_avg_sq,
                    )
                else:
                    optim_result = self.hyper_map(
                        ops.partial(
                            _optim_adamw_opt,
                            self.adamw_opt,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            state_step,
                            self.amsgrad,
                            self.maximize,
                            lr,
                        ),
                        weight_decay,
                        self._master_parameters,
                        gradients,
                        self.exp_avg,
                        self.exp_avg_sq,
                        self.max_exp_avg_sq,
                    )
            else:
                optim_result = self.hyper_map(
                    ops.partial(
                        _optim_adamw_opt,
                        self.adamw_opt,
                        self.beta1,
                        self.beta2,
                        self.eps,
                        state_step,
                        self.amsgrad,
                        self.maximize,
                        lr,
                        weight_decay,
                    ),
                    self._master_parameters,
                    gradients,
                    self.exp_avg,
                    self.exp_avg_sq,
                    self.max_exp_avg_sq,
                )
        else:
            if self.is_group:
                if self.is_group_lr:
                    optim_result = self.hyper_map(
                        ops.partial(
                            _optim_adamw_opt,
                            self.adamw_opt,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            state_step,
                            self.amsgrad,
                            self.maximize,
                        ),
                        lr,
                        weight_decay,
                        self._master_parameters,
                        gradients,
                        self.exp_avg,
                        self.exp_avg_sq,
                    )
                else:
                    optim_result = self.hyper_map(
                        ops.partial(
                            _optim_adamw_opt,
                            self.adamw_opt,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            state_step,
                            self.amsgrad,
                            self.maximize,
                            lr,
                        ),
                        weight_decay,
                        self._master_parameters,
                        gradients,
                        self.exp_avg,
                        self.exp_avg_sq,
                    )
            else:
                optim_result = self.hyper_map(
                    ops.partial(
                        _optim_adamw_opt,
                        self.adamw_opt,
                        self.beta1,
                        self.beta2,
                        self.eps,
                        state_step,
                        self.amsgrad,
                        self.maximize,
                        lr,
                        weight_decay,
                    ),
                    self._master_parameters,
                    gradients,
                    self.exp_avg,
                    self.exp_avg_sq,
                )

        optim_result = ops.depend(optim_result, self.map_(update_, self._master_parameters, self._parameters))
        return optim_result
