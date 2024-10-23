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
import mindspore.ops as ops

from .samplers import create_sampler


class Sampler(object):
    def __init__(self, sampler_name, sd_model, steps, cfg_scale):
        super().__init__()
        self.sampler_name = sampler_name
        self.sd_model = sd_model
        self.sampler = create_sampler(sampler_name, sd_model)
        self.steps = steps
        self.cfg_scale = cfg_scale

    def sample(
        self,
        S,
        conditioning,
        batch_size,
        shape,
        verbose,
        unconditional_guidance_scale,
        unconditional_conditioning,
        eta,
        x_T,
    ):
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f"Data shape for {self.sampler_name} sampling is {size}")
        if x_T is None:
            img = ops.standard_normal(size)
        else:
            img = x_T
        x = img  # ms.Tensor
        samples = self.sampler.sample(
            self, x, conditioning, unconditional_conditioning, image_conditioning=x.new_zeros(x.shape[0], dtype=x.dtype)
        )
        return samples, None
