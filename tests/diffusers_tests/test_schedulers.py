import importlib
import inspect
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
import torch

import mindspore as ms
import mindspore.nn as nn

from mindone.diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

STEP_THR_FP16 = 1e-2
STEP_THR_FP32 = 1e-4
THR = 1e-5


class NoiseSchedulerWrapper(nn.Cell):
    def __init__(self, ns):
        super(NoiseSchedulerWrapper, self).__init__()
        self.ns = ns

    def construct(self, method_name, *args, **kwargs):
        if method_name == "add_noise":
            return self.ns.add_noise(*args, **kwargs)
        elif method_name == "get_velocity":
            return self.ns.get_velocity(*args, **kwargs)
        elif method_name == "scale_model_input":
            return self.ns.scale_model_input(*args, **kwargs)
        elif method_name == "step":
            return self.ns.step(*args, **kwargs)
        else:
            raise NotImplementedError


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]
    dtype = torch.float32 if dtype == ms.float32 else torch.float16

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return ms.Tensor(latents.numpy())


def grab_all_schedulers():
    karras_schedulers = {scheduler.name for scheduler in KarrasDiffusionSchedulers}
    maybe_schedulers = inspect.getmembers(sys.modules["mindone.diffusers.schedulers"], inspect.isclass)
    schedulers = filter(
        lambda x: issubclass(x[1], SchedulerMixin) and x[0] in karras_schedulers and x[0] != "SchedulerMixin",
        maybe_schedulers,
    )
    return schedulers


@pytest.mark.parametrize("scheduler_name", grab_all_schedulers())
@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_schedulers(scheduler_name, dtype):
    # TODO: Add different conditions
    ms.set_context(mode=ms.PYNATIVE_MODE)
    # set input
    np.random.seed(0)
    noise = np.random.randn(4, 3, 8, 8)
    scheduler_name_pt, scheduler_name_ms = scheduler_name
    # set dtype
    ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
    # replace randn_tensor
    sys.modules[scheduler_name_ms.__module__].randn_tensor = randn_tensor
    # set_timesteps & ms
    scheduler_ms = scheduler_name_ms()
    scheduler_ms.set_timesteps(50)
    t_start = scheduler_ms.num_inference_steps - 2
    timestep_ms = scheduler_ms.timesteps[0]
    timesteps_ms = scheduler_ms.timesteps[t_start * scheduler_ms.order :]
    scheduler_ms = NoiseSchedulerWrapper(scheduler_ms)
    # set_timesteps & pt
    scheduler_pt_modules = importlib.import_module("diffusers.schedulers")
    scheduler_pt = getattr(scheduler_pt_modules, scheduler_name_pt)()
    scheduler_pt.set_timesteps(50)
    t_start = scheduler_pt.num_inference_steps - 2
    timestep_pt = scheduler_pt.timesteps[0]
    timesteps_pt = scheduler_pt.timesteps[t_start * scheduler_pt.order :]
    # add_noise
    if hasattr(scheduler_pt, "add_noise"):
        output_ms = scheduler_ms("add_noise", ms.tensor(noise, ms_dtype), ms.tensor(noise, ms_dtype), timesteps_ms[:1])
        output_pt = scheduler_pt.add_noise(
            torch.tensor(noise, dtype=pt_dtype), torch.tensor(noise, dtype=pt_dtype), timesteps_pt[:1]
        )
        assert np.max(np.abs(output_ms.asnumpy() - output_pt.numpy())) / np.mean(np.abs(output_pt.numpy())) < THR
        assert output_ms.dtype == ms_dtype and output_pt.dtype == pt_dtype
    # get_velocity
    if hasattr(scheduler_pt, "get_velocity"):
        output_ms = scheduler_ms(
            "get_velocity", ms.tensor(noise, ms_dtype), ms.tensor(noise, ms_dtype), timesteps_ms[:1]
        )
        output_pt = scheduler_pt.get_velocity(
            torch.tensor(noise, dtype=pt_dtype), torch.tensor(noise, dtype=pt_dtype), timesteps_pt[:1]
        )
        assert np.max(np.abs(output_ms.asnumpy() - output_pt.numpy())) / np.mean(np.abs(output_pt.numpy())) < THR
        assert output_ms.dtype == ms_dtype and output_pt.dtype == pt_dtype
    # scale_model_input
    output_ms = scheduler_ms("scale_model_input", ms.tensor(noise, ms_dtype), timesteps_ms[0])
    output_pt = scheduler_pt.scale_model_input(torch.tensor(noise, dtype=pt_dtype), timesteps_pt[0])
    assert np.max(np.abs(output_ms.asnumpy() - output_pt.numpy())) / np.mean(np.abs(output_pt.numpy())) < THR
    assert output_ms.dtype == ms_dtype and output_pt.dtype == pt_dtype
    # step
    torch.manual_seed(0)
    output_ms = scheduler_ms(
        "step", model_output=ms.tensor(noise, ms_dtype), timestep=timestep_ms, sample=ms.tensor(noise, ms_dtype)
    )[0]
    torch.manual_seed(0)
    output_pt = scheduler_pt.step(
        model_output=torch.tensor(noise, dtype=pt_dtype),
        timestep=timestep_pt,
        sample=torch.tensor(noise, dtype=pt_dtype),
        return_dict=False,
    )[0]
    if dtype == "float16":
        assert (
            np.max(np.abs(output_ms.asnumpy() - output_pt.numpy())) / np.mean(np.abs(output_pt.numpy())) < STEP_THR_FP16
        )
    else:
        assert (
            np.max(np.abs(output_ms.asnumpy() - output_pt.numpy())) / np.mean(np.abs(output_pt.numpy())) < STEP_THR_FP32
        )
    assert output_ms.dtype == ms_dtype and output_pt.dtype == pt_dtype
