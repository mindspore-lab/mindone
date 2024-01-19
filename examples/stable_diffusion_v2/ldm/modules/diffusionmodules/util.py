# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer


class Identity(nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()
        self.identity = ops.Identity()

    def construct(self, x):
        return self.identity(x)


def linear(in_channel, out_channel, dtype=ms.float32):
    """
    Create a linear module.
    """
    return nn.Dense(in_channel, out_channel).to_float(dtype)


class conv_nd(nn.Cell):
    def __init__(self, dims, *args, **kwargs):
        super().__init__()
        if dims == 1:
            self.conv = nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            self.conv = nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            self.conv = nn.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")

    def construct(self, x, emb=None, context=None):
        x = self.conv(x)
        return x


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    weight = initializer("zeros", module.conv.weight.shape)
    bias_weight = initializer("zeros", module.conv.bias.shape)
    module.conv.weight.set_data(weight)
    module.conv.bias.set_data(bias_weight)

    return module


class avg_pool_nd(nn.Cell):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """

    def __init__(self, dims, *args, **kwargs):
        super().__init__()
        if dims == 1:
            self.avgpool = nn.AvgPool1d(*args, **kwargs)
        elif dims == 2:
            self.avgpool = nn.AvgPool2d(*args, **kwargs)
        elif dims == 3:
            self.avgpool = ops.AvgPool3D(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")

    def construct(self, x, emb=None, context=None):
        x = self.avgpool(x)
        return x


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Cell for normalization.
    """
    return GroupNorm32(32, channels).to_float(ms.float32)


class SiLU(nn.Cell):
    def __init__(self, upcast=False):
        super(SiLU, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.upcast = upcast

    def construct(self, x):
        if self.upcast:
            # force sigmoid to use fp32
            return x * self.sigmoid(x.astype(ms.float32)).astype(x.dtype)
        else:
            return x * self.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def construct(self, x):
        # return ops.cast(super().construct(ops.cast(x, ms.float32)), x.dtype)
        return super().construct(x)


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = ops.exp(
            -ops.log(ms.Tensor(max_period, ms.float32)) * ms.numpy.arange(start=0, stop=half, dtype=ms.float32) / half
        )
        args = timesteps[:, None] * freqs[None]
        embedding = ops.concat((ops.cos(args), ops.sin(args)), axis=-1)
        if dim % 2:
            embedding = ops.concat((embedding, ops.ZerosLike()(embedding[:, :1])), axis=-1)
    else:
        embedding = ops.reshape(timesteps.repeat(dim), (-1, dim))
    return embedding


def make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=50, num_ddpm_timesteps=1000, verbose=False):
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = ms.Tensor(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f"Selected timesteps for ddim sampler: {steps_out}")
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta=0.0, verbose=False):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = ops.concat((ms.numpy.array([alphacums[0]]), alphacums[ddim_timesteps[:-1]]))

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * ops.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f"Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}")
        print(
            f"For the chosen value of eta, which is {eta}, "
            f"this results in the following sigma_t schedule for ddim sampler {sigmas}"
        )
    return sigmas, alphas, alphas_prev


def noise_like(shape, repeat=False):
    if not repeat:
        return ms.ops.StandardNormal()(shape)
    else:
        raise ValueError("The repeat method is not supported")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def make_beta_schedule(schedule="linear", n_timestep=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        start = linear_start**0.5
        stop = linear_end**0.5
        num = n_timestep
        betas = (np.linspace(start, stop, num) ** 2).astype(np.float32)
    elif schedule == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            n_timestep,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")

    return betas
