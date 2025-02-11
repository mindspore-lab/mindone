# Copyright 2023 Huawei Technologies Co., Ltd
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
import logging

import mindspore as ms
from mindspore import ops

from .dpm_solver import DPM_Solver, NoiseScheduleVP, model_wrapper

_logger = logging.getLogger(__name__)


class DPMSolverSampler(object):
    def __init__(self, model, algorithm_type="dpmsolver", **kwargs):
        super().__init__()
        self.model = model
        self.algorithm_type = algorithm_type
        self.register_buffer("alphas_cumprod", model.alphas_cumprod)
        self.noise_schedule = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)
        self.prediction_type = kwargs.get("prediction_type", "noise")

    def register_buffer(self, name, attr):
        setattr(self, name, attr)

    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        skip_type="time_uniform",
        method="multistep",
        order=2,
        lower_order_final=True,
        correcting_xt_fn=None,
        t_start=None,
        t_end=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    _logger.warning(f"Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    _logger.warning(f"Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        size = (batch_size, *shape)
        _logger.debug(
            f"Data shape for DPM-Solver sampling is {size}"
        ) if self.algorithm_type == "dpmsolver" else _logger.debug(f"Data shape for DPM-Solver++ sampling is {size}")

        if x_T is None:
            img = ops.standard_normal(size)
        else:
            img = x_T

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            self.noise_schedule,
            model_type=self.prediction_type,
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        dpm_solver = DPM_Solver(
            model_fn, self.noise_schedule, algorithm_type=self.algorithm_type, correcting_xt_fn=correcting_xt_fn
        )

        x, intermediates = dpm_solver.sample(
            img,
            steps=S,
            t_start=t_start,
            t_end=t_end,
            skip_type=skip_type,
            method=method,
            order=order,
            lower_order_final=lower_order_final,
            return_intermediate=True,
        )

        return x, intermediates

    def stochastic_encode(self, x0, encode_ratio, noise=None):
        t_end = self.ratio_to_time(encode_ratio)
        t_end = ops.Cast()(ms.Tensor([t_end]), x0.dtype)
        x = DPM_Solver(None, self.noise_schedule).add_noise(x0, t_end, noise=noise)
        return x

    def encode(
        self,
        S,
        x,
        encode_ratio,
        conditioning=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        skip_type="time_uniform",
        method="multistep",
        order=2,
        lower_order_final=False,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != x.shape[0]:
                    _logger.warning(f"Got {cbs} conditionings but batch-size is {x.shape[0]}")
            else:
                if conditioning.shape[0] != x.shape[0]:
                    _logger.warning(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {x.shape[0]}"
                    )

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            self.noise_schedule,
            model_type=self.prediction_type,
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        t_end = self.ratio_to_time(encode_ratio)

        dpm_solver = DPM_Solver(model_fn, self.noise_schedule, algorithm_type=self.algorithm_type)

        x, intermediates = dpm_solver.inverse(
            x,
            steps=S,
            t_end=t_end,
            skip_type=skip_type,
            method=method,
            order=order,
            lower_order_final=lower_order_final,
            return_intermediate=True,
        )

        return x, intermediates

    def time_discrete_to_continuous(self, t_discrete):
        """
        Convert [0, 999] to [0.001, 1].
        """
        return (t_discrete + 1.0) / self.noise_schedule.total_N

    def time_continuous_to_discrete(self, t_continuous):
        """
        Convert [0.001, 1] to [0, 999].
        """
        return t_continuous * self.noise_schedule.total_N - 1.0

    def ratio_to_time(self, ratio):
        """
        Convert [0, 1] to [0.001, 1].
        """
        return (1.0 - 1.0 / self.noise_schedule.total_N) * ratio + 1.0 / self.noise_schedule.total_N

    def time_to_ratio(self, t_continuous):
        """
        Convert [0.001, 1] to [0, 1].
        """
        return (t_continuous - 1.0 / self.noise_schedule.total_N) / (1.0 - self.noise_schedule.total_N)
