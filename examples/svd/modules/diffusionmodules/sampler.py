from gm.modules.diffusionmodules.sampler import EulerEDMSampler as EulerEDMSamplerOriginal

from mindspore import dtype as ms_dtype


class EulerEDMSampler(EulerEDMSamplerOriginal):
    def denoise(self, x, model, sigma, cond, uc, **kwargs):
        num_frames = kwargs["num_frames"]

        noised_input, sigmas, cond = self.guider.prepare_inputs(x, sigma, cond, uc)
        cond = model.openai_input_warpper(cond)
        c_skip, c_out, c_in, c_noise = model.denoiser(sigmas, noised_input.ndim)
        model_output = model.model(noised_input * c_in, c_noise, **cond, **kwargs)
        model_output = model_output.astype(ms_dtype.float32)
        denoised = model_output * c_out + noised_input * c_skip
        denoised = self.guider(denoised, sigma, num_frames)
        return denoised
