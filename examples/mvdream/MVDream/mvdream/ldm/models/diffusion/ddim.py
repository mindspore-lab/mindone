"""SAMPLING ONLY. No need for the no_grad decorator, as the sampler won't be implemented as a child class of nn.Cell"""

import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor, mint, ops

from ...modules.diffusionmodules.util import (
    extract_into_tensor,
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", img_eps=None, **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.img_eps = img_eps

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"

        setattr(self, "betas", self.model.betas)
        # setattr(self, 'alphas_cumprod', to_npu(alphas_cumprod))  # this already on npu/cuda, why need this?
        setattr(self, "alphas_cumprod", alphas_cumprod)
        setattr(self, "alphas_cumprod_prev", self.model.alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        setattr(self, "sqrt_alphas_cumprod", np.sqrt(alphas_cumprod.asnumpy()))
        setattr(self, "sqrt_one_minus_alphas_cumprod", np.sqrt(1.0 - alphas_cumprod.asnumpy()))
        # setattr(self, 'log_one_minus_alphas_cumprod', to_npu(np.log(1. - alphas_cumprod.asnumpy())))
        # setattr(self, 'sqrt_recip_alphas_cumprod', to_npu(np.sqrt(1. / alphas_cumprod.asnumpy())))
        # setattr(self, 'sqrt_recipm1_alphas_cumprod', to_npu(np.sqrt(1. / alphas_cumprod.asnumpy() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.asnumpy(), ddim_timesteps=self.ddim_timesteps, eta=ddim_eta, verbose=verbose
        )
        setattr(self, "ddim_sigmas", ddim_sigmas)
        setattr(self, "ddim_alphas", ddim_alphas)
        setattr(self, "ddim_alphas_prev", ddim_alphas_prev)
        setattr(self, "ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * mint.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        setattr(self, "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps)

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

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            **kwargs,
        )
        return samples, intermediates

    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        **kwargs,
    ):
        b = shape[0]
        if x_T is None:
            # img = ops.randn(shape)
            img = Tensor(self.img_eps).to(ms.float32)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = mint.full((b,), int(step), dtype=ms.int32)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img

            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )
            img, pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
        **kwargs,
    ):
        b, *_ = x.shape

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = mint.cat([x] * 2)
            t_in = mint.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [mint.cat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
                    elif isinstance(c[k], Tensor):
                        c_in[k] = mint.cat([unconditional_conditioning[k], c[k]])
                    else:
                        assert c[k] == unconditional_conditioning[k]
                        c_in[k] = c[k]
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(mint.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = mint.cat([unconditional_conditioning, c])

            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            # model_t = self.model.apply_model(x, t, c, **kwargs)
            # model_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            print("using v!")
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", "not implemented"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = mint.full((b, 1, 1, 1), Tensor(alphas[index]))
        a_prev = mint.full((b, 1, 1, 1), Tensor(alphas_prev[index]))
        sigma_t = mint.full((b, 1, 1, 1), Tensor(sigmas[index]))
        sqrt_one_minus_at = mint.full((b, 1, 1, 1), Tensor(sqrt_one_minus_alphas[index]))

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = mint.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = mint.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = ops.randn_like(x0)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def decode(
        self,
        x_latent,
        cond,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_original_steps=False,
        **kwargs,
    ):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc="Decoding image", total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = mint.full((x_latent.shape[0],), step, dtype=ms.int32)
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )
        return x_dec
