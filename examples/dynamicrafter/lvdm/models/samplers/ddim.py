import numpy as np
from lvdm.modules.networks.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, rescale_noise_cfg
from tqdm import tqdm

import mindspore as ms
import mindspore.ops as ops

from mindone.utils.misc import extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.split = ops.Split(0, 2)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"

        if self.model.use_dynamic_rescale:
            self.ddim_scale_arr = self.model.scale_arr[self.ddim_timesteps]
            self.ddim_scale_arr_prev = ops.cat([self.ddim_scale_arr[0:1], self.ddim_scale_arr[:-1]])

        self.betas = self.model.betas
        self.alphas_cumprod = self.model.alphas_cumprod
        self.alphas_cumprod_prev = self.model.alphas_cumprod_prev

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = ops.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = ops.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = ops.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = ops.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = ops.sqrt(1.0 / alphas_cumprod - 1)

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod, ddim_timesteps=self.ddim_timesteps, eta=ddim_eta, verbose=verbose
        )

        self.ddim_sigmas = ddim_sigmas
        self.ddim_alphas = ddim_alphas
        self.ddim_alphas_prev = ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = ops.sqrt(1.0 - ddim_alphas)
        sigmas_for_original_sampling_steps = ddim_eta * ops.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.ddim_sigmas_for_original_num_steps = sigmas_for_original_sampling_steps

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
        precision=None,
        fs=None,
        timestep_spacing="uniform",  # uniform_trailing for starting from last timestep
        guidance_rescale=0.0,
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_discretize=timestep_spacing, ddim_eta=eta, verbose=verbose)

        # sampling
        size = (batch_size, *shape)
        print(f"Data shape for DDIM sampling is {size}, eta {eta}")
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
            verbose=verbose,
            precision=precision,
            fs=fs,
            guidance_rescale=guidance_rescale,
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
        verbose=True,
        precision=None,
        fs=None,
        guidance_rescale=0.0,
        **kwargs,
    ):
        b = shape[0]
        if x_T is None:
            img = ms.Tensor(np.random.randn(*shape), dtype=ms.float32)
        else:
            img = x_T
        if precision is not None:
            if precision == 16:
                img = img.to(dtype=ms.float16)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else ms.numpy.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = time_range
        clean_cond = kwargs.pop("clean_cond", False)

        for i, step in tqdm(enumerate(iterator), desc="DDIM Sampling", total=len(iterator)):
            index = total_steps - i - 1
            ts = ms.numpy.full((b,), step, dtype=ms.int64)

            if mask is not None:
                assert x0 is not None
                if clean_cond:
                    img_orig = x0
                else:
                    img_orig = self.model.q_sample(x0, ts)
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
                mask=mask,
                x0=x0,
                fs=fs,
                guidance_rescale=guidance_rescale,
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
        uc_type=None,
        conditional_guidance_scale_temporal=None,
        mask=None,
        x0=None,
        guidance_rescale=0.0,
        **kwargs,
    ):
        b = x.shape[0]
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False

        for k, v in c.items():
            if isinstance(v, list):
                assert len(v) == 1
                c[k] = v[0]

        for k, v in unconditional_conditioning.items():
            if isinstance(v, list):
                assert len(v) == 1
                unconditional_conditioning[k] = v[0]

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            model_output = self.model.apply_model(x, t, c, **kwargs)  # unet denoiser
        else:
            # do_classifier_free_guidance
            if isinstance(c, ms.Tensor) or isinstance(c, dict):
                e_t_cond = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            else:
                raise NotImplementedError

            model_output = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)

            if guidance_rescale > 0.0:
                model_output = rescale_noise_cfg(model_output, e_t_cond, guidance_rescale=guidance_rescale)

        if self.model.parameterization == "velocity":
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
        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = ms.numpy.full(size, alphas[index])
        a_prev = ms.numpy.full(size, alphas_prev[index])
        sigma_t = ms.numpy.full(size, sigmas[index])
        sqrt_one_minus_at = ms.numpy.full(size, sqrt_one_minus_alphas[index])

        # current prediction for x_0
        if self.model.parameterization != "velocity":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if self.model.use_dynamic_rescale:
            scale_t = ops.full(size, self.ddim_scale_arr[index], dtype=self.ddim_scale_arr[index].dtype)
            prev_scale_t = ops.full(size, self.ddim_scale_arr_prev[index], dtype=self.ddim_scale_arr_prev[index].dtype)
            rescale = prev_scale_t / scale_t
            pred_x0 *= rescale

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise, _ = ops.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

    def encode(
        self,
        x0,
        c,
        t_enc,
        use_original_steps=False,
        return_intermediates=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        callback=None,
    ):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = self.ddim_alphas_prev[:num_steps]

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:num_steps]
        iterator = tqdm(timesteps, desc="Encoding image", total=timesteps.shape[0])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i, step in enumerate(iterator):
            t = ms.numpy.full((x0.shape[0],), step, dtype=ms.int64)
            if unconditional_guidance_scale == 1.0:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond = self.model.apply_model(x_next, t, unconditional_conditioning)
                noise_pred = self.model.apply_model(x_next, t, c)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = (
                alphas_next[i].sqrt() * ((1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            )
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback:
                callback(i)

        out = {"x_encoded": x_next, "intermediate_steps": inter_steps}
        if return_intermediates:
            out.update({"intermediates": intermediates})
        return x_next, out

    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = ops.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = ms.numpy.randn(x0.shape)
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
        callback=None,
    ):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="Decoding image", total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = ms.numpy.full((x_latent.shape[0],), step, dtype=ms.int64)
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
            if callback:
                callback(i)
        return x_dec
