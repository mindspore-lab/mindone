"""Modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py"""
import math
from typing import List, Optional, Tuple, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.ops as ops
from mindspore import Parameter, ParameterTuple, Tensor
from mindspore.experimental.optim.optimizer import Optimizer

_muon_opt = ops.MultitypeFuncGraph("muon_opt")


@_muon_opt.register(
    "Float",
    "Float",
    "Float",
    "Float",
    "Bool",
    "Int",
    "Float",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Float",
    "Bool",
)
def _update_run_op(
    mu: float,
    beta1: float,
    beta2: float,
    eps: float,
    nesterov: bool,
    ns_steps: int,
    weight_decay: float,
    lr: Parameter,
    denom: Parameter,
    param: Parameter,
    m: Parameter,
    v: Parameter,
    g: Tensor,
    ratio: float,
    use_muon: bool,
) -> bool:
    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)

    if use_muon:
        m.mul_(mu).add_(g)
        if nesterov:
            g = g.add(m, alpha=mu)
        else:
            g = m
        g = zeropower_via_newtonschulz5(g, steps=ns_steps)
        param.add_(lr * g, alpha=-ratio)
    else:
        m_next = mint.lerp(g, m, beta1)
        v_next = mint.lerp(mint.square(g), v, beta2)
        g = m_next / (eps + mint.sqrt(v_next))
        param.add_(-(lr / denom) * g)
        ops.assign(m, m_next)
        ops.assign(v, v_next)
    return True


_qk_clip_opt = ops.MultitypeFuncGraph("qk_clip_opt")


@_qk_clip_opt.register("Float", "Int", "Tensor", "Tensor", "Tensor")
def _update_clip_op(
    clip_value: float, qk_nope_head_dim: int, qk: Tensor, q_b_projs: Parameter, kv_b_projs: Parameter
) -> bool:
    qk = mint.transpose(qk, 0, 1).flatten(start_dim=1)
    qk_max, _ = mint.max(qk, dim=1)
    num_head = qk_max.shape[0]
    scale = mint.clip(clip_value / qk_max, max=1.0)
    scale = scale[:, None, None]
    scale_sqrt = mint.sqrt(scale)
    # clip Q projection
    outdim, _ = q_b_projs.shape
    head_dim = outdim // num_head
    scale_q_b_nope = mint.tile(scale_sqrt, (1, qk_nope_head_dim, 1))
    scale_q_b_rope = mint.tile(scale, (1, head_dim - qk_nope_head_dim, 1))
    scale_q_b = mint.cat([scale_q_b_nope, scale_q_b_rope], dim=1)
    q_b_projs.mul_(scale_q_b.view(-1, 1))
    # clip K projection
    outdim, _ = kv_b_projs.shape
    head_dim = outdim // num_head
    scale_kv_b_nope = mint.tile(scale_sqrt, (1, qk_nope_head_dim, 1))
    scale_kv_b_rope = mint.ones((num_head, head_dim - qk_nope_head_dim, 1), dtype=scale_sqrt.dtype)
    scale_kv_b = mint.cat([scale_kv_b_nope, scale_kv_b_rope], dim=1)
    kv_b_projs.mul_(scale_kv_b.view(-1, 1))
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

    if len(shape) > 2:
        G = G.view(G.shape[0], -1)
    assert len(shape) == 2

    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if G.shape[0] > G.shape[1]:
        X = mint.t(X)

    # Ensure spectral norm is at most 1
    X = X / (mint.norm(X) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = mint.matmul(X, X.T)
        B = mint.addmm(A, A, A, beta=b, alpha=c)  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = mint.addmm(X, B, X, beta=a)

    if G.shape[0] > G.shape[1]:
        X = mint.t(X)

    if len(shape) > 2:
        X = X.view(*shape)
    return X


class Muon(Optimizer):
    """Following https://github.com/MoonshotAI/Moonlight"""

    def __init__(
        self,
        lr: Union[float, Tensor] = 1e-3,
        wd: float = 0.1,
        muon_params: Optional[List[Parameter]] = None,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_params: Optional[List[Parameter]] = None,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        clip_value: Optional[float] = 100.0,
        qk_nope_head_dim: int = 64,
    ) -> None:
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        self.clip_value = clip_value
        self.qk_nope_head_dim = qk_nope_head_dim
        # Sort parameters into those for which we will use Muon, and those for which we will not
        use_muon = list()
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim >= 2, p.ndim
            use_muon.append(True)

        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            use_muon.append(False)
        self.use_muon = tuple(use_muon)

        self.exp_avg = self.parameters.clone("exp_avg", init="zeros")
        self.exp_avg_sq = ParameterTuple(
            [
                (
                    Parameter(mint.zeros(x.shape, dtype=x.dtype), name="exp_avg_sq." + x.name)
                    if not use_muon
                    else Parameter([], name="exp_avg_sq." + x.name)
                )
                for x, use_muon in zip(self.parameters, self.use_muon)
            ]
        )

        self.lr_ratio = tuple([self._cal_lr_ratio(x, use_muon) for x, use_muon in zip(self.parameters, self.use_muon)])

        self.state_step = Parameter(Tensor(0, dtype=ms.int32))
        self.increase_tensor = Tensor(1, dtype=ms.int32)
        self.denom = Parameter(Tensor(1.0, dtype=ms.float32))

        if self.clip_value is not None:
            # group the Q and KV projection first for easier updating in QK-clip
            # TODO: it should be extracted from optimizer as extra inputs
            q_b_projs = []
            kv_b_projs = []
            for x in self.parameters:
                if x.name.endswith("q_b_proj.weight"):
                    layer_idx = int(x.name.split(".")[2])
                    q_b_projs.append((layer_idx, x))
                elif x.name.endswith("kv_b_proj.weight"):
                    layer_idx = int(x.name.split(".")[2])
                    kv_b_projs.append((layer_idx, x))
            q_b_projs = sorted(q_b_projs, key=lambda x: x[0])
            kv_b_projs = sorted(kv_b_projs, key=lambda x: x[0])
            self.q_b_projs = ParameterTuple([x[1] for x in q_b_projs])
            self.kv_b_projs = ParameterTuple([x[1] for x in kv_b_projs])
            assert len(self.q_b_projs) > 0 and len(self.kv_b_projs) > 0

    def _cal_lr_ratio(self, param: Parameter, use_muon: bool, rms_scale: float = 0.2) -> float:
        if not use_muon:
            return 1.0

        A, B = param.shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = rms_scale * math.sqrt(max(A, B))
        return adjusted_ratio

    @ms.jit(jit_level="O1")
    def muon(
        self,
        momentum: float,
        beta1: float,
        beta2: float,
        eps: float,
        nesterov: bool,
        ns_steps: int,
        weight_decay: float,
        lr: Parameter,
        gradients: Tuple[Tensor, ...],
        ratio: Tuple[float, ...],
        use_muon: Tuple[bool, ...],
        start_id: int,
        end_id: int,
    ) -> bool:
        bias_correction1 = 1 - beta1**self.state_step
        bias_correction2 = 1 - beta2**self.state_step
        ops.assign(self.denom, bias_correction1 / bias_correction2**0.5)

        optim_result = self.hyper_map(
            ops.partial(
                _muon_opt,
                momentum,
                beta1,
                beta2,
                eps,
                nesterov,
                ns_steps,
                weight_decay,
                lr,
                self.denom,
            ),
            self.parameters[start_id:end_id],
            self.exp_avg[start_id:end_id],
            self.exp_avg_sq[start_id:end_id],
            gradients[start_id:end_id],
            ratio[start_id:end_id],
            use_muon[start_id:end_id],
        )
        return optim_result

    @ms.jit(jit_level="O1")
    def qk_clip(self, qk_products: Tuple[Tensor, ...]) -> bool:
        optim_result = self.hyper_map(
            ops.partial(_qk_clip_opt, self.clip_value, self.qk_nope_head_dim),
            qk_products,
            self.q_b_projs,
            self.kv_b_projs,
        )
        return optim_result

    def construct(self, gradients: Tuple[Tensor, ...], qk_products: Optional[Tuple[Tensor, ...]] = None) -> bool:
        if self.clip_value is not None:
            assert qk_products is not None

        self.state_step.add_(self.increase_tensor)
        for group_id, group in enumerate(self.param_groups):
            beta1, beta2 = group["adamw_betas"]
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]

            self.muon(
                group["momentum"],
                beta1,
                beta2,
                group["adamw_eps"],
                group["nesterov"],
                group["ns_steps"],
                group["weight_decay"],
                group["lr"],
                gradients,
                self.lr_ratio,
                self.use_muon,
                start_id,
                end_id,
            )

        if self.clip_value is None:
            return True
        else:
            optim_result = self.qk_clip(qk_products)
            return optim_result
