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
from mindspore import ops

from .uni_pc import NoiseScheduleVP, UniPC, model_wrapper


class UniPCSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.register_buffer("alphas_cumprod", model.alphas_cumprod)
        self.noise_schedule = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)

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
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        size = (batch_size, *shape)
        print(f"Data shape for UniPC sampling is {size}")

        if x_T is None:
            img = ops.standard_normal(size)
        else:
            img = x_T

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            self.noise_schedule,
            model_type="noise",
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        uni_pc = UniPC(model_fn, self.noise_schedule, predict_x0=True, thresholding=False)

        x = uni_pc.sample(img, steps=S, skip_type="time_uniform", method="multistep", order=3, lower_order_final=True)

        return x, None
