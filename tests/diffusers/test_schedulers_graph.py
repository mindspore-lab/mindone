import importlib
import inspect
import sys

import numpy as np
import pytest
import torch

import mindspore as ms
import mindspore.nn as nn

from mindone.diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

THR_FP16 = 1e-2  # On the NPU, you can lower the threshold.
THR_FP32 = 1e-5


class NoiseSchedulerWrapper(nn.Cell):
    def __init__(self, ns):
        super(NoiseSchedulerWrapper, self).__init__()
        self.ns = ns

    def construct(self, method_name, *args, **kwargs):
        if method_name == "add_noise":
            return self.ns.add_noise(*args, **kwargs)
        elif method_name == "get_velocity":
            return self.ns.get_velocity(*args, **kwargs)
        else:
            raise NotImplementedError


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
    ms.set_context(mode=ms.GRAPH_MODE)
    # set input
    np.random.seed(0)
    noise = np.random.randn(4, 3, 8, 8)
    scheduler_name_pt, scheduler_name_ms = scheduler_name
    # set dtype
    ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
    # set_timesteps & ms
    scheduler_ms = scheduler_name_ms()
    scheduler_ms.set_timesteps(50)
    t_start = scheduler_ms.num_inference_steps - 2
    timesteps_ms = scheduler_ms.timesteps[t_start * scheduler_ms.order :]
    scheduler_ms = NoiseSchedulerWrapper(scheduler_ms)
    # set_timesteps & pt
    scheduler_pt_modules = importlib.import_module("diffusers.schedulers")
    scheduler_pt = getattr(scheduler_pt_modules, scheduler_name_pt)()
    scheduler_pt.set_timesteps(50)
    t_start = scheduler_pt.num_inference_steps - 2
    timesteps_pt = scheduler_pt.timesteps[t_start * scheduler_pt.order :]
    # add_noise
    if hasattr(scheduler_pt, "add_noise"):
        output_ms = scheduler_ms("add_noise", ms.tensor(noise, ms_dtype), ms.tensor(noise, ms_dtype), timesteps_ms[:1])
        output_pt = scheduler_pt.add_noise(
            torch.tensor(noise, dtype=pt_dtype), torch.tensor(noise, dtype=pt_dtype), timesteps_pt[:1]
        )
        if dtype == "float16":
            assert (
                np.max(np.abs(output_ms.asnumpy() - output_pt.numpy())) / np.mean(np.abs(output_pt.numpy())) < THR_FP16
            )
        else:
            assert (
                np.max(np.abs(output_ms.asnumpy() - output_pt.numpy())) / np.mean(np.abs(output_pt.numpy())) < THR_FP32
            )
        assert output_ms.dtype == ms_dtype and output_pt.dtype == pt_dtype
    # get_velocity
    if hasattr(scheduler_pt, "get_velocity"):
        output_ms = scheduler_ms(
            "get_velocity", ms.tensor(noise, ms_dtype), ms.tensor(noise, ms_dtype), timesteps_ms[:1]
        )
        output_pt = scheduler_pt.get_velocity(
            torch.tensor(noise, dtype=pt_dtype), torch.tensor(noise, dtype=pt_dtype), timesteps_pt[:1]
        )
        assert np.max(np.abs(output_ms.asnumpy() - output_pt.numpy())) / np.mean(np.abs(output_pt.numpy())) < THR_FP32
        assert output_ms.dtype == ms_dtype and output_pt.dtype == pt_dtype
