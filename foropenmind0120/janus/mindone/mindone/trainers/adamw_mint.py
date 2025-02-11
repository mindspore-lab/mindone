from __future__ import absolute_import

import mindspore as ms
from mindspore import _checkparam as validator
from mindspore import ops
from mindspore.nn import Optimizer
from mindspore.ops import auto_generate as gen

_optim_adamw_opt = ops.MultitypeFuncGraph("optim_adamw_opt")
hyper_map = ops.HyperMap()


@_optim_adamw_opt.register(
    "Function",
    "Float",
    "Float",
    "Float",
    "Tensor",
    "Bool",
    "Bool",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
)
def _run_optim_adamw_opt(
    opt, beta1, beta2, eps, step, amsgrad, maximize, learning_rate, weight_decay, parameters, grads, exp_avg, exp_avg_sq
):
    """Apply adamw optimizer to the weight parameter."""
    success = True
    max_exp_avg_sq = ops.zeros_like(exp_avg)
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    opt(
        parameters,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        grads.astype(parameters.dtype),
        step,
        learning_rate,
        beta1,
        beta2,
        weight_decay,
        eps,
        amsgrad,
        maximize,
    )
    return success


@_optim_adamw_opt.register(
    "Function",
    "Float",
    "Float",
    "Float",
    "Tensor",
    "Bool",
    "Bool",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
)
def _run_optim_adamw_amsgrad_opt(
    opt,
    beta1,
    beta2,
    eps,
    step,
    amsgrad,
    maximize,
    learning_rate,
    weight_decay,
    parameters,
    grads,
    exp_avg,
    exp_avg_sq,
    max_exp_avg_sq,
):
    """Apply adamw optimizer to the weight parameter."""
    success = True
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    opt(
        parameters,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        grads.astype(parameters.dtype),
        step,
        learning_rate,
        beta1,
        beta2,
        weight_decay,
        eps,
        amsgrad,
        maximize,
    )
    return success


class AdamW(Optimizer):
    r"""
    Implements Adam Weight Decay algorithm.

    .. math::
        \begin{aligned}
            &\textbf{input}      : \gamma \text{(learning_rate)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
       \end{aligned}

    .. warning::
        This is an experimental optimizer API that is subject to change.
        This module must be used with learning_rate scheduler module in `learning_rateScheduler Class
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#learning_ratescheduler-class>`_ .

    Args:
        params (Union[list(Parameter), list(dict)]): list of parameters to optimize or dicts defining
            parameter groups
        learning_rate (Union[int, float, Tensor], optional): learning rate. Default: ``1e-3``.
        beta1 (float, optional): Coefficients used for computing running averages of gradient.
            Default: ``0.9``.
        beta2 (float, optional): Coefficients used for computing running averages of gradient.
            Default: ``0.999``.
        eps (float, optional): term added to the denominator to improve
            numerical stability. Default: ``1e-6``.
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``0.``.
        amsgrad (bool, optional): whether to use the AMSGrad algorithm. Default: ``False``.

    Keyword Args:
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: ``False``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the `eps` is less than 0.0.
        ValueError: If the `betas` not in the range of [0, 1).
        ValueError: If the `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend``
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
        super().__init__(learning_rate, params, weight_decay)
        self._check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.exp_avg = self.parameters.clone(prefix="exp_avg", init="zeros")
        self.exp_avg_sq = self.parameters.clone(prefix="exp_avg_sq", init="zeros")
        if amsgrad:
            self.max_exp_avg_sq = self.parameters.clone(prefix="max_exp_avg_sq", init="zeros")
        self.adamw_opt = gen.AdamW()
        self.amsgrad = amsgrad
        self.maximize = maximize

    def _check_param_value(self, beta1, beta2, eps, prim_name):
        """Check the type of inputs."""
        validator.check_value_type("beta1", beta1, [float], prim_name)
        validator.check_value_type("beta2", beta2, [float], prim_name)
        validator.check_value_type("eps", eps, [float], prim_name)
        validator.check_float_range(beta1, 0.0, 1.0, validator.INC_NEITHER, "beta1", prim_name)
        validator.check_float_range(beta2, 0.0, 1.0, validator.INC_NEITHER, "beta2", prim_name)
        validator.check_positive_float(eps, "eps", prim_name)

    def construct(self, gradients):
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)
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
                        self._parameters,
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
                        self._parameters,
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
                    self._parameters,
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
                        self._parameters,
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
                        self._parameters,
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
                    self._parameters,
                    gradients,
                    self.exp_avg,
                    self.exp_avg_sq,
                )
        return optim_result
