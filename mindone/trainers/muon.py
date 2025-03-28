import logging
import math
from typing import Callable, List, Optional, Tuple

import numpy as np

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, ParameterTuple, Tensor
from mindspore.communication import GlobalComm, get_group_size, get_rank

_logger = logging.getLogger(__name__)

_muon_opt = ops.MultitypeFuncGraph("muon_opt")


@_muon_opt.register(
    "Number",
    "Number",
    "Number",
    "Tensor",
    "Tensor",
    "Number",
    "Bool",
    "Number",
    "Function",
    "Number",
    "Number",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Number",
    "Bool",
    "Bool",
    "Bool",
    "Bool",
)
def _update_run_op(
    mu: float,
    beta1: float,
    beta2: float,
    beta1_t: Parameter,
    beta2_t: Parameter,
    eps: float,
    nesterov: bool,
    steps: int,
    allgather: Callable[[Tensor], Tensor],
    rank_id: int,
    group_size: int,
    lr: Parameter,
    weight_decay: Tensor,
    param: Parameter,
    m: Parameter,
    v: Parameter,
    g: Tensor,
    ratio: float,
    use_muon: bool,
    decay_flag: bool,
    optim_filter: bool,
    parallel_optimizer: bool,
) -> bool:
    if not optim_filter:
        return False

    if decay_flag:
        param.add_(-lr * weight_decay * param)

    v_next = None
    if use_muon:
        # Muon branch
        m_next = mint.lerp(g, m, mu)
        if nesterov:
            g = mint.lerp(g, m_next, mu)
        else:
            g = m_next
        if parallel_optimizer:
            g = _allgather_along_first_dim(g, allgather)
        g = zeropower_via_newtonschulz5(g, steps=steps)
        if parallel_optimizer:
            g = _split_along_first_dim(g, rank_id, group_size)
        param.add_(-lr * ratio * g)
    else:
        # AdamW branch
        m_next = mint.lerp(g, m, beta1)
        v_next = mint.lerp(mint.square(g), v, beta2)
        m_hat = m_next / (1 - beta1_t)
        v_hat = v_next / (1 - beta2_t)
        g = m_hat / (mint.sqrt(v_hat) + eps)
        param.add_(-lr * g)

    ops.assign(m, m_next)
    if not use_muon:
        ops.assign(v, v_next)
    return True


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    shape = G.shape
    dtype = G.dtype
    assert len(shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    G = G.bfloat16()

    if len(shape) > 2:
        G = mint.reshape(G, (G.shape[0], -1))

    need_transpose = G.shape[0] > G.shape[1]
    if need_transpose:
        G = G.T
    # Ensure spectral norm is at most 1
    G = G / (mint.norm(G) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = G @ G.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        G = a * G + B @ G

    if need_transpose:
        G = G.T

    if len(shape) > 2:
        G = mint.reshape(G, shape)

    return G.to(dtype)


def _allgather_along_first_dim(x: Tensor, allgather: Callable[[Tensor], Tensor]) -> Tensor:
    x = allgather(x)
    return x


def _split_along_first_dim(x: Tensor, rank_id: int, group_size: int) -> Tensor:
    chunk_size = x.shape[0] // group_size
    x = mint.narrow(x, 0, chunk_size * rank_id, chunk_size)
    return x


class Muon(nn.Optimizer):
    """Following https://github.com/MoonshotAI/Moonlight"""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adamw_betas: Tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        adamw_parameter_names: Optional[Tuple[str, ...]] = ("embed_tokens", "lm_head"),
        rms_scale: float = 0.2,
        optimizer_parallel_group: Optional[str] = None,
    ) -> None:
        super().__init__(lr, params, weight_decay)

        if not isinstance(adamw_parameter_names, (tuple, list)):
            raise ValueError("`adamw_parameter_names` must be a tuple or list.")
        if adamw_parameter_names is None:
            adamw_parameter_names = tuple([])

        self.momentum = momentum
        self.adamw_beta1 = adamw_betas[0]
        self.adamw_beta2 = adamw_betas[1]
        self.adamw_eps = adamw_eps
        self.moments1 = ParameterTuple(
            [Parameter(np.zeros(x.shape, dtype=np.float32), name="m." + x.name) for x in self._parameters]
        )
        self.use_muon = tuple([self._use_muon(x, adamw_parameter_names) for x in self._parameters])
        self.moments2 = ParameterTuple(
            [
                (
                    Parameter(np.zeros(x.shape, dtype=np.float32), name="v." + x.name)
                    if not use_muon
                    else Parameter([], name="v." + x.name)
                )
                for x, use_muon in zip(self._parameters, self.use_muon)
            ]
        )
        self.adamw_beta1_t = Parameter(Tensor(1, dtype=ms.float32))
        self.adamw_beta2_t = Parameter(Tensor(1, dtype=ms.float32))
        self.ns_steps = ns_steps
        self.nesterov = nesterov

        self.lr_ratio = tuple(
            [
                self._cal_lr_ratio(x, use_muon, rms_scale=rms_scale)
                for x, use_muon in zip(self._parameters, self.use_muon)
            ]
        )

        self.optimizer_parallel_group = optimizer_parallel_group
        if self.optimizer_parallel_group is not None and GlobalComm.INITED:
            self.allgather = ops.AllGather(group=self.optimizer_parallel_group)
            self.group_size = get_group_size(self.optimizer_parallel_group)
            self.rank_id = get_rank(self.optimizer_parallel_group)
            self.enable_distributed_mode()
        else:
            self.allgather = ops.Identity()
            self.group_size = 1
            self.rank_id = 0
            self.disable_distributed_mode()

    def refresh_parallel_optimizer_states(self):
        """Update the parallel optimizer state for each parameter.
        It should be called after parameter split in ZeRO.
        """
        self.parallel_optimizer_states = tuple([x.parallel_optimizer and GlobalComm.INITED for x in self._parameters])

    def disable_distributed_mode(self):
        _logger.debug("Disable distributed mode for Muon optimizer.")
        for x in self._parameters:
            x.parallel_optimizer = False
        self.refresh_parallel_optimizer_states()

    def enable_distributed_mode(self):
        _logger.debug("Enable distributed mode for Muon optimizer.")
        for x in self._parameters:
            x.parallel_optimizer = True
        self.refresh_parallel_optimizer_states()

    def _use_muon(self, param: Parameter, adamw_parameter_names: Tuple[str, ...]) -> bool:
        for name in adamw_parameter_names:
            if name in param.name:
                return False
        if len(param.shape) >= 2:
            if "weight" not in param.name:
                _logger.warning(
                    f"Get unusual parameter under Muon optimizer group with name `{param.name}` and shape `{param.shape}`, "
                    "perhaps you need to add the parameter name in the argument of `adamw_parameter_names`."
                )
            return True
        return False

    def _cal_lr_ratio(self, param: Parameter, use_muon: bool, rms_scale: float = 0.2) -> float:
        if not use_muon:
            return 1.0

        A, B = param.shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = rms_scale * math.sqrt(max(A, B))
        return adjusted_ratio

    @ms.jit
    def construct(self, gradients: List[Tensor]) -> bool:
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        self.adamw_beta1_t = self.adamw_beta1_t * self.adamw_beta1
        self.adamw_beta2_t = self.adamw_beta2_t * self.adamw_beta2

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    ops.partial(
                        _muon_opt,
                        self.momentum,
                        self.adamw_beta1,
                        self.adamw_beta2,
                        self.adamw_beta1_t,
                        self.adamw_beta2_t,
                        self.adamw_eps,
                        self.nesterov,
                        self.ns_steps,
                        self.allgather,
                        self.rank_id,
                        self.group_size,
                    ),
                    lr,
                    weight_decay,
                    self._parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.lr_ratio,
                    self.use_muon,
                    self.decay_flags,
                    self.optim_filter,
                    self.parallel_optimizer_states,
                )
            else:
                optim_result = self.hyper_map(
                    ops.partial(
                        _muon_opt,
                        self.momentum,
                        self.adamw_beta1,
                        self.adamw_beta2,
                        self.adamw_beta1_t,
                        self.adamw_beta2_t,
                        self.adamw_eps,
                        self.nesterov,
                        self.ns_steps,
                        self.allgather,
                        self.rank_id,
                        self.group_size,
                        lr,
                    ),
                    weight_decay,
                    self._parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.lr_ratio,
                    self.use_muon,
                    self.decay_flags,
                    self.optim_filter,
                    self.parallel_optimizer_states,
                )
        else:
            optim_result = self.hyper_map(
                ops.partial(
                    _muon_opt,
                    self.momentum,
                    self.adamw_beta1,
                    self.adamw_beta2,
                    self.adamw_beta1_t,
                    self.adamw_beta2_t,
                    self.adamw_eps,
                    self.nesterov,
                    self.ns_steps,
                    self.allgather,
                    self.rank_id,
                    self.group_size,
                    lr,
                    weight_decay,
                ),
                self._parameters,
                self.moments1,
                self.moments2,
                gradients,
                self.lr_ratio,
                self.use_muon,
                self.decay_flags,
                self.optim_filter,
                self.parallel_optimizer_states,
            )
        return optim_result
