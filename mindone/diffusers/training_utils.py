import random

import numpy as np

import mindspore as ms
from mindspore import ops
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


def multinomial_rand(p: ms.Tensor, size: tuple):
    assert isinstance(p, ms.Tensor) and p.ndim == 1, "Probability p should be a 1-dim MindSpore tensor."

    p = p.float()
    p /= p.sum()
    p = p.cumsum()
    for _ in size:
        p = p.expand_dims(axis=0)

    rand = ops.rand(*size, dtype=p.dtype).expand_dims(-1)
    multinomial_rand = ops.ge(rand, p).float().sum(axis=-1).long()

    return multinomial_rand
