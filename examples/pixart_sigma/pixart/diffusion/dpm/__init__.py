from typing import Any, Callable, Dict

from mindspore import Tensor

from ..iddpm import get_named_beta_schedule

# from .dpm_solver import DPM_Solver, NoiseScheduleVP, model_wrapper
from .dpm_solver_np import DPM_Solver, NoiseScheduleVP, model_wrapper


def DPMS(
    model: Callable[..., Tensor],
    noise_schedule: NoiseScheduleVP,
    condition: Tensor,
    uncondition: Tensor,
    cfg_scale: float,
    model_type: str = "noise",  # or "x_start" or "v" or "score"
    guidance_type: str = "classifier-free",
    model_kwargs: Dict[str, Any] = {},
) -> DPM_Solver:
    # 1. Convert your discrete-time `model` to the continuous-time
    # noise prediction model. Here is an example for a diffusion model
    # `model` with the noise prediction type ("noise") .
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type=model_type,
        model_kwargs=model_kwargs,
        guidance_type=guidance_type,
        condition=condition,
        unconditional_condition=uncondition,
        guidance_scale=cfg_scale,
    )
    # 3. Define dpm-solver and sample by multistep DPM-Solver.
    return DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")


def create_noise_schedule_dpms(noise_schedule: str = "linear", diffusion_steps: int = 1000) -> NoiseScheduleVP:
    betas = Tensor(get_named_beta_schedule(noise_schedule, diffusion_steps))
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=betas)
    return noise_schedule
