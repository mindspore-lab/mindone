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


import inspect
from collections import namedtuple

import mindspore as ms
import mindspore.ops as ops

from ..sampler import external, sampling

SamplerData = namedtuple("SamplerData", ["name", "constructor", "aliases", "options"])
samplers_k_diffusion = [
    ("Euler a", "sample_euler_ancestral", ["k_euler_a", "k_euler_ancestral"], {"uses_ensd": True}),
    ("Euler", "sample_euler", ["k_euler"], {}),
    ("LMS", "sample_lms", ["k_lms"], {}),
    ("Heun", "sample_heun", ["k_heun"], {"second_order": True}),
    ("DPM2", "sample_dpm_2", ["k_dpm_2"], {"discard_next_to_last_sigma": True}),
    ("DPM2 a", "sample_dpm_2_ancestral", ["k_dpm_2_a"], {"discard_next_to_last_sigma": True, "uses_ensd": True}),
    ("DPM++ 2S a", "sample_dpmpp_2s_ancestral", ["k_dpmpp_2s_a"], {"uses_ensd": True, "second_order": True}),
    ("DPM++ 2M", "sample_dpmpp_2m", ["k_dpmpp_2m"], {}),
    ("DPM fast", "sample_dpm_fast", ["k_dpm_fast"], {"uses_ensd": True}),
    ("DPM adaptive", "sample_dpm_adaptive", ["k_dpm_ad"], {"uses_ensd": True}),
    ("LMS Karras", "sample_lms", ["k_lms_ka"], {"scheduler": "sampler"}),
    (
        "DPM2 Karras",
        "sample_dpm_2",
        ["k_dpm_2_ka"],
        {"scheduler": "sampler", "discard_next_to_last_sigma": True, "uses_ensd": True, "second_order": True},
    ),
    (
        "DPM2 a Karras",
        "sample_dpm_2_ancestral",
        ["k_dpm_2_a_ka"],
        {"scheduler": "sampler", "discard_next_to_last_sigma": True, "uses_ensd": True, "second_order": True},
    ),
    (
        "DPM++ 2S a Karras",
        "sample_dpmpp_2s_ancestral",
        ["k_dpmpp_2s_a_ka"],
        {"scheduler": "sampler", "uses_ensd": True, "second_order": True},
    ),
    ("DPM++ 2M Karras", "sample_dpmpp_2m", ["k_dpmpp_2m_ka"], {"scheduler": "sampler"}),
]
samplers_data_k_diffusion = [
    SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
    for label, funcname, aliases, options in samplers_k_diffusion
    if hasattr(sampling, funcname)
]
sampler_extra_params = {
    "sample_euler": ["s_churn", "s_tmin", "s_tmax", "s_noise"],
    "sample_heun": ["s_churn", "s_tmin", "s_tmax", "s_noise"],
    "sample_dpm_2": ["s_churn", "s_tmin", "s_tmax", "s_noise"],
}


def catenate_conds(conds):
    return ops.cat(conds)


def subscript_cond(cond, a, b):
    return cond[a:b]


def pad_cond(tensor, repeats, empty):
    if not isinstance(tensor, dict):
        return ops.cat([tensor, empty.repeat((tensor.shape[0], repeats, 1))], axis=1)

    tensor["crossattn"] = pad_cond(tensor["crossattn"], repeats, empty)
    return tensor


class CFGDenoiser(object):
    """
    Classifier free guidance denoiser. A wrapper for stable diffusion model (specifically for unet)
    that can take a noisy picture and produce a noise-free picture using two guidances (prompts)
    instead of one. Originally, the second prompt is just an empty string, but we use non-empty
    negative prompt.
    """

    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.step = 0
        self.image_cfg_scale = None
        self.padded_cond_uncond = False

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0] :]
        denoised = ms.Tensor(denoised_uncond)

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)

        return denoised

    def combine_denoised_for_edit_model(self, x_out, cond_scale):
        out_cond, out_img_cond, out_uncond = x_out.chunk(3)
        denoised = (
            out_uncond + cond_scale * (out_cond - out_img_cond) + self.image_cfg_scale * (out_img_cond - out_uncond)
        )

        return denoised

    def __call__(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
        is_edit_model = False

        batch_size = cond.shape[0]
        conds_list = []
        for i in range(batch_size):
            conds_list.append([(i, 1.0)])
        repeats = [len(conds_list[i]) for i in range(batch_size)]

        image_uncond = image_cond
        if isinstance(uncond, dict):
            make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, "c_concat": [c_concat]}
        else:
            make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": [c_crossattn], "c_concat": [c_concat]}

        x_in = ops.cat([ops.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
        sigma_in = ops.cat([ops.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])
        image_cond_in = ops.cat(
            [ops.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond]
        )
        text_cond = cond
        skip_uncond = False

        # alternating uncond allows for higher thresholds without the quality loss normally expected from raising it
        if self.step % 2 and s_min_uncond > 0 and sigma[0] < s_min_uncond and not is_edit_model:
            skip_uncond = True
            x_in = x_in[:-batch_size]
            sigma_in = sigma_in[:-batch_size]

        if text_cond.shape[1] == uncond.shape[1] or skip_uncond:
            if is_edit_model:
                cond_in = catenate_conds([text_cond, uncond, uncond])
            elif skip_uncond:
                cond_in = text_cond
            else:
                cond_in = catenate_conds([text_cond, uncond])
            x_out = self.inner_model(x_in, sigma_in, c_crossattn=cond_in)

        else:
            x_out = ops.zeros_like(x_in)
            batch_size = batch_size * 2
            for batch_offset in range(0, text_cond.shape[0], batch_size):
                a = batch_offset
                b = min(a + batch_size, text_cond.shape[0])

                if not is_edit_model:
                    c_crossattn = subscript_cond(text_cond, a, b)
                else:
                    c_crossattn = ops.cat([text_cond[a:b]], uncond)

                x_out[a:b] = self.inner_model(
                    x_in[a:b], sigma_in[a:b], cond=make_condition_dict(c_crossattn, image_cond_in[a:b])
                )

            if not skip_uncond:
                x_out[-uncond.shape[0] :] = self.inner_model(
                    x_in[-uncond.shape[0] :],
                    sigma_in[-uncond.shape[0] :],
                    cond=make_condition_dict(uncond, image_cond_in[-uncond.shape[0] :]),
                )

        denoised_image_indexes = [x[0][0] for x in conds_list]
        if skip_uncond:
            fake_uncond = ops.cat([x_out[i : i + 1] for i in denoised_image_indexes])
            x_out = ops.cat(
                [x_out, fake_uncond]
            )  # we skipped uncond denoising, so we put cond-denoised image to where the uncond-denoised image should be

        if skip_uncond:
            denoised = self.combine_denoised(x_out, conds_list, uncond, 1.0)
        else:
            denoised = self.combine_denoised(x_out, conds_list, uncond, cond_scale)

        if self.mask is not None:
            denoised = self.init_latent * self.mask + self.nmask * denoised
        self.step += 1
        return denoised


class KDiffusionSampler:
    def __init__(self, funcname, sd_model):
        denoiser = external.CompVisVDenoiser if sd_model.parameterization == "v" else external.CompVisDenoiser

        self.model_wrap = denoiser(sd_model)
        self.funcname = funcname
        self.func = getattr(sampling, self.funcname)
        self.extra_params = sampler_extra_params.get(funcname, [])
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.sampler_noises = None
        self.stop_at = None
        self.eta = None
        self.config = None  # set by the function calling the constructor
        self.last_latent = None
        self.s_min_uncond = None

        self.conditioning_key = sd_model.model.conditioning_key

    def launch_sampling(self, func):
        return func()

    def initialize(self, p):
        self.model_wrap_cfg.mask = p.mask if hasattr(p, "mask") else None
        self.model_wrap_cfg.nmask = p.nmask if hasattr(p, "nmask") else None
        self.model_wrap_cfg.step = 0
        self.model_wrap_cfg.image_cfg_scale = getattr(p, "image_cfg_scale", None)
        self.eta = 1.0
        self.s_min_uncond = getattr(p, "s_min_uncond", 0.0)

        extra_params_kwargs = {}
        for param_name in self.extra_params:
            if hasattr(p, param_name) and param_name in inspect.signature(self.func).parameters:
                extra_params_kwargs[param_name] = getattr(p, param_name)

        if "eta" in inspect.signature(self.func).parameters:
            if self.eta != 1.0:
                p.extra_generation_params["Eta"] = self.eta

            extra_params_kwargs["eta"] = self.eta

        return extra_params_kwargs

    def get_sigmas(self, p, steps):
        discard_next_to_last_sigma = self.config is not None and self.config.options.get(
            "discard_next_to_last_sigma", False
        )
        steps += 1 if discard_next_to_last_sigma else 0

        if self.config is not None and self.config.options.get("scheduler", None) == "sampler":
            sigma_min, sigma_max = (self.model_wrap.sigmas[0], self.model_wrap.sigmas[-1])

            sigmas = sampling.get_sigmas_karras(n=steps, sigma_min=sigma_min, sigma_max=sigma_max)
        else:
            sigmas = self.model_wrap.get_sigmas(steps)

        if discard_next_to_last_sigma:
            sigmas = ops.cat([sigmas[:-2], sigmas[-1:]])

        return sigmas

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        steps = steps or p.steps

        sigmas = self.get_sigmas(p, steps)

        x = x * sigmas[0]

        extra_params_kwargs = self.initialize(p)

        parameters = inspect.signature(self.func).parameters

        if "sigma_min" in parameters:
            extra_params_kwargs["sigma_min"] = self.model_wrap.sigmas[0]
            extra_params_kwargs["sigma_max"] = self.model_wrap.sigmas[-1]
            if "n" in parameters:
                extra_params_kwargs["n"] = steps
        else:
            extra_params_kwargs["sigmas"] = sigmas
        self.last_latent = x
        samples = self.launch_sampling(
            lambda: self.func(
                self.model_wrap_cfg,
                x,
                extra_args={
                    "cond": conditioning,
                    "image_cond": image_conditioning,
                    "uncond": unconditional_conditioning,
                    "cond_scale": p.cfg_scale,
                    "s_min_uncond": self.s_min_uncond,
                },
                disable=False,
                **extra_params_kwargs
            )
        )
        return samples
