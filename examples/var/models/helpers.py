# This code is adapted from https://github.com/FoundationVision/VAR
# with modifications to run on MindSpore.

import mindspore as ms
import mindspore.nn.probability.distribution as msd
from mindspore import mint, nn, ops


def sample_with_top_k_top_p_(
    logits_BlV: ms.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1
) -> ms.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        idx_to_remove = logits_BlV < mint.amin(
            logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0], -1, keepdim=True
        )

        logits_BlV.masked_fill_(idx_to_remove, -ms.numpy.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = mint.softmax(sorted_logits, dim=-1).cumsum(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(
            sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -ms.numpy.inf
        )
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return mint.multinomial(
        mint.softmax(logits_BlV, dim=-1).view((-1, V)), num_samples=num_samples, replacement=replacement, generator=rng
    ).view((B, l, num_samples))


def gumbel_softmax_with_rng(
    logits: ms.Tensor, tau: float = 1, hard: bool = False, dim: int = -1, rng=None
) -> ms.Tensor:
    if rng is None:
        return ops.gumbel_softmax(logits=logits, tau=tau, hard=hard, dim=dim)

    # gumbels = (
    #     -mint.empty_like(logits).exponential_(generator=rng).log())
    e1 = msd.Exponential(1, seed=rng.initial_seed())
    gumbels = -e1.sample(logits.shape).log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = mint.zeros_like(logits).scatter(dim, index, 1.0)
        ret = y_hard - y_soft + y_soft
    else:
        ret = y_soft
    return ret


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):  # taken from timm
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    random_tensor = ops.bernoulli(ops.zeros(shape), p=keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div(keep_prob)
    return x * random_tensor


class DropPath(nn.Cell):  # taken from timm
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
