from ldm.models.diffusion.ddim import DDIMSampler as BaseDDIMSampler
import numpy as np
from tqdm import tqdm
import mindspore as ms

from .p2p import register_attention_control


class DDIMSampler(BaseDDIMSampler):
    def __init__(self, model, schedule="linear", controller=None):
        super().__init__(model, schedule)
        self.controller = controller
        register_attention_control(model, controller)

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
