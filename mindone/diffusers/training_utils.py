import random

import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore.common.api import _function_forbid_reuse
from mindspore.communication import get_group_size, get_local_rank, get_rank, init


def set_seed(seed=42, rank=0):
    ms.set_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def is_master(args):
    return args.rank == 0


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if args.distributed:
        init()
        args.local_rank = get_local_rank()
        args.world_size = get_group_size()
        args.rank = get_rank()
        ms.context.set_auto_parallel_context(
            device_num=args.world_size,
            global_rank=args.rank,
            parallel_mode="data_parallel",
            gradients_mean=True,
        )

    device = f"{ms.get_context('device_target')}:{ms.get_context('device_id')}"
    args.device = device
    return device


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026  # noqa: E501
    # we do not expand alpha/sigma which is redundant for the broadcast shape is actually timesteps.shape
    alpha = sqrt_alphas_cumprod[timesteps].float()
    sigma = sqrt_one_minus_alphas_cumprod[timesteps].float()

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


@_function_forbid_reuse
def multinomial(input, num_samples, replacement=True, **kwargs):
    assert isinstance(input, ms.Tensor) and input.ndim in (
        1,
        2,
    ), "argument input should be a MindSpore Tensor with 1 or 2 dim."
    assert (
        replacement or num_samples <= input.shape[-1]
    ), "cannot sample n_sample > prob_dist.size(-1) samples without replacement."

    input = input.float()
    input /= input.sum(-1, keepdims=True)

    if num_samples == 1 or not replacement:
        # The algorithm is from gumbel softmax.
        # s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1)
        # Here we can apply exp to the formula which will not affect result of
        # argmax or topk. Then we have
        # s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
        # We can also simplify the formula above by
        # s = argmax( p / q ) where q ~ Exp(1)
        # No proper Exp generator op in MindSpore,
        # so we still generate it by -log(eps)
        q = -ops.log(ops.rand_like(input))
        if num_samples == 1:
            result = (input / q).argmax(-1, keepdim=True)
        else:
            _, result = ops.topk(input / q, k=num_samples, dim=-1)
    else:
        # To generate scalar random variable X with cumulative distribution F(x)
        # just let X = F^(-1)(U) where U ~ U(0, 1)
        input = input.cumsum(-1).expand_dims(-1)
        rshape = (1, num_samples) if input.ndim == 2 else (input.shape[0], 1, num_samples)
        rand = ops.rand(*rshape, dtype=input.dtype)
        result = ops.ge(rand, input).long().sum(-2)

    return result.long()
