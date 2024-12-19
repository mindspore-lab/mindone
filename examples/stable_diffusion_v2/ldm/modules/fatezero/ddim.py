import numpy as np
from ldm.models.diffusion.ddim import DDIMSampler as BaseDDIMSampler
from tqdm import tqdm

import mindspore as ms

from ..diffusionmodules.util import noise_like
from .p2p import register_attention_control


class DDIMSampler(BaseDDIMSampler):
    def __init__(self, model, schedule="linear", controller=None):
        super().__init__(model, schedule)
        self.edit_controller = None
        self.controller = controller
        register_attention_control(model, controller)

    def pre_sample(self, model, controller=None):
        self.controller.is_invert = False
        self.edit_controller = controller
        register_attention_control(model, controller)

        self.edit_controller.attention_store_all_step = self.controller.attention_store_all_step
        self.edit_controller.pos_dict = self.controller.pos_dict

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
        features_adapter=None,
        append_to_context=None,
    ):
        b = x.shape[0]
        print("index:", index)
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            c_in = c if append_to_context is None else ms.ops.cat([c, append_to_context], axis=1)
            model_output = self.model.apply_model(x, t, c_in, features_adapter=features_adapter)
        else:
            x_in = ms.ops.concat((x, x), axis=0)
            t_in = ms.ops.concat((t, t), axis=0)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [
                            ms.ops.concat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]), axis=0)
                        ]
                    else:
                        c_in[k] = ms.ops.concat([unconditional_conditioning[k], c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(ms.ops.concat([unconditional_conditioning[i], c[i]], axis=0))
            else:
                if append_to_context is not None:
                    pad_len = append_to_context.size(1)
                    new_unconditional_conditioning = ms.ops.cat(
                        [unconditional_conditioning, unconditional_conditioning[:, -pad_len:, :]], axis=1
                    )
                    new_c = ms.ops.cat([c, append_to_context], axis=1)
                    c_in = ms.ops.cat([new_unconditional_conditioning, new_c])
                else:
                    c_in = ms.ops.concat([unconditional_conditioning, c], axis=0)
            model_uncond, model_t = self.split(
                self.model.apply_model(x_in, t_in, c_in, features_adapter=features_adapter)
            )
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
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
        shape = (1,) * (len(e_t.shape) - 1)
        a_t = ms.numpy.full((b,) + shape, alphas[index])
        a_prev = ms.numpy.full((b,) + shape, alphas_prev[index])
        sigma_t = ms.numpy.full((b,) + shape, sigmas[index])
        sqrt_one_minus_at = ms.numpy.full((b,) + shape, sqrt_one_minus_alphas[index])

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
            noise, _ = ms.ops.dropout(noise, p=noise_dropout)
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
            if self.controller:
                x_next = self.controller.step_callback(x_next)
        out = {"x_encoded": x_next, "intermediate_steps": inter_steps}
        if return_intermediates:
            out.update({"intermediates": intermediates})
        return x_next, out
