import random

import numpy as np

import mindspore as ms
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
    sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr
