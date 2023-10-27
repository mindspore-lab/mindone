from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import tqdm

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor

from ..utils.export import model_export
from .ddim import DDIM
from .ddpm import DDPM
from .plms import PLMS

__all__ = ["beta_schedule", "DiffusionSampler"]


def beta_schedule(
    schedule: str, num_timesteps: int = 1000, init_beta: Optional[float] = None, last_beta: Optional[float] = None
) -> np.ndarray:
    """This code defines a function beta_schedule that generates a sequence of beta values based on
    the given input parameters. These beta values can be used in video diffusion processes.

    Args:
        schedule(str): Determines the type of beta schedule to be generated.
            It can be 'linear', 'linear_sd', 'quadratic', or 'cosine'.
        num_timesteps(int, optional): The number of timesteps for the generated beta schedule. Default is 1000.
        init_beta(float, optional): The initial beta value.
            If not provided, a default value is used based on the chosen schedule.
        last_beta(float, optional): The final beta value.
            If not provided, a default value is used based on the chosen schedule.

    Returns:
        The function returns a PyTorch tensor containing the generated beta values.
        The beta schedule is determined by the schedule parameter:
        1.Linear: Generates a linear sequence of beta values between init_beta and last_beta.
        2.Linear_sd: Generates a linear sequence of beta values between the square root of init_beta and
            the square root of last_beta, and then squares the result.
        3.Quadratic: Similar to the 'linear_sd' schedule, but with different default values for init_beta and last_beta.
        4.Cosine: Generates a sequence of beta values based on a cosine function,
            ensuring the values are between 0 and 0.999.

    Raises:
        If an unsupported schedule is provided, a ValueError is raised with a message indicating the issue.
    """
    if schedule == "linear":
        scale = 1000.0 / num_timesteps
        init_beta = init_beta or scale * 0.0001
        last_beta = last_beta or scale * 0.02
        return np.linspace(init_beta, last_beta, num_timesteps)
    elif schedule == "linear_sd":
        return np.linspace(init_beta**0.5, last_beta**0.5, num_timesteps) ** 2
    elif schedule == "quadratic":
        init_beta = init_beta or 0.0015
        last_beta = last_beta or 0.0195
        return np.linspace(init_beta**0.5, last_beta**0.5, num_timesteps) ** 2
    elif schedule == "cosine":
        betas = []
        for step in range(num_timesteps):
            t1 = step / num_timesteps
            t2 = (step + 1) / num_timesteps
            fn = lambda u: np.cos((u + 0.008) / 1.008 * np.pi / 2) ** 2
            betas.append(min(1.0 - fn(t2) / fn(t1), 0.999))
        return np.array(betas)
    else:
        raise ValueError(f"Unsupported schedule: {schedule}")


class DiffusionSampler:
    """A Diffusion Sampler wrapper that performs a unet forward and a scheduler sampling for n steps."""

    def __init__(
        self,
        model: nn.Cell,
        scheduler_name: str = "DDIM",
        betas: Optional[np.ndarray] = None,
        num_timesteps: int = 1000,
        show_progress_bar: bool = True,
    ):
        if betas is None:
            betas = beta_schedule("linear_sd", num_timesteps, init_beta=0.00085, last_beta=0.0120)

        if scheduler_name == "DDIM":
            self.scheduler = DDIM(model, betas).set_train(False)
        elif scheduler_name == "PLMS":
            self.scheduler = PLMS(model, betas).set_train(False)
        elif scheduler_name == "DDPM":
            self.scheduler = DDPM(model, betas).set_train(False)
        else:
            raise NotImplementedError(f"Scheduler `{scheduler_name}` is not supported.")
        for _, param in self.scheduler.parameters_and_names():
            param.requires_grad = False

        self.num_timesteps = len(betas)
        self.show_progress_bar = show_progress_bar
        self.scheduler_name = scheduler_name

    def __call__(
        self,
        noise: Tensor,
        model_kwargs: Dict[str, Tensor] = {},
        guide_scale: float = 9.0,
        timesteps: int = 20,
        eta: float = 0.0,
        export_only: bool = False,
        export_name: str = "scheduler",
    ) -> Tensor:
        xt = noise

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = 1 + np.arange(0, self.num_timesteps, self.num_timesteps // timesteps)
        steps = np.clip(steps, 0, self.num_timesteps - 1)
        steps = np.flip(steps).astype(np.int32)

        stride = self.num_timesteps // timesteps

        # need to pass the full inputs list of the unet here for export.
        # need to check the args in unet_sd.py
        model_full_args = OrderedDict(
            y=None,
            depth=None,
            image=None,
            motion=None,
            local_image=None,
            single_sketch=None,
            masked=None,
            canny=None,
            sketch=None,
            fps=None,
            video_mask=None,
            focus_present_mask=None,
            prob_focus_present=0.0,
            mask_last_frame_num=0,
        )

        stride = Tensor(stride, dtype=ms.int32)
        eta = Tensor(eta, dtype=ms.float32)
        guide_scale = Tensor(guide_scale, dtype=ms.float32)
        for k, v in model_kwargs.items():
            model_full_args[k] = v

        for step in tqdm.tqdm(steps, desc="sample_loop", disable=not self.show_progress_bar, leave=False):
            if self.scheduler_name == "DDPM":
                xt = self.scheduler(xt, Tensor(step), guide_scale, *model_full_args.values())
            elif self.scheduler_name == "PLMS":
                xt = self.scheduler(xt, Tensor(step), stride, guide_scale, *model_full_args.values())
            else:
                xt = self.scheduler(xt, Tensor(step), stride, eta, guide_scale, *model_full_args.values())

            if export_only:
                if self.scheduler_name == "DDPM":
                    model_export(
                        self.scheduler,
                        [xt, Tensor(step, ms.int32), guide_scale, *model_full_args.values()],
                        export_name,
                    )
                elif self.scheduler_name == "PLMS":
                    model_export(
                        self.scheduler,
                        [xt, Tensor(step, ms.int32), stride, guide_scale, *model_full_args.values()],
                        export_name,
                    )
                else:
                    model_export(
                        self.scheduler,
                        [xt, Tensor(step, ms.int32), stride, eta, guide_scale, *model_full_args.values()],
                        export_name,
                    )
                break

        if self.scheduler_name == "PLMS":
            # TODO: on MS2.1 + 910B, it does not work. Uncomment this after fix.
            # self.scheduler.reset_cache()
            pass

        return xt
