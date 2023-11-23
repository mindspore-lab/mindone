from abc import abstractmethod

import mindspore_lite as mslite
import numpy as np
from tqdm import tqdm

from .model_base import ModelBase


class SDLite(ModelBase):
    def __init__(
        self,
        data_prepare,
        scheduler_preprocess,
        predict_noise,
        noisy_sample,
        vae_decoder,
        denoiser,
        scheduler_prepare_sampling_loop,
        device_target="ascend",
        device_id=0,
        num_inference_steps=50,
    ):
        super(SDLite, self).__init__(device_target, device_id)
        self.data_prepare = self._init_model(data_prepare)
        self.scheduler_preprocess = self._init_model(scheduler_preprocess)
        self.predict_noise = self._init_model(predict_noise)
        self.noisy_sample = self._init_model(noisy_sample)
        self.vae_decoder = self._init_model(vae_decoder)
        self.denoiser = self._init_model(denoiser)
        self.scheduler_prepare_sampling_loop = self._init_model(scheduler_prepare_sampling_loop)

        n_infer_steps = mslite.Tensor()
        n_infer_steps.shape = []
        n_infer_steps.dtype = mslite.DataType.INT32
        n_infer_steps.set_data_from_numpy(np.array(num_inference_steps, np.int32))
        self.num_inference_steps = n_infer_steps

        # get input
        self.data_prepare_input = self.data_prepare.get_inputs()
        self.scheduler_preprocess_input = self.scheduler_preprocess.get_inputs()
        self.predict_noise_input = self.predict_noise.get_inputs()
        self.noisy_sample_input = self.noisy_sample.get_inputs()
        self.vae_decoder_input = self.vae_decoder.get_inputs()
        self.denoiser_input = self.denoiser.get_inputs()

    @abstractmethod
    def data_prepare_predict(self, inputs):
        pass

    def __call__(self, inputs):
        scale = inputs["scale"]
        context, y, noise = self.data_prepare_predict(inputs)
        x, s_in = self.scheduler_prepare_sampling_loop.predict([noise])
        for i in tqdm(range(inputs["timesteps"]), desc="SDXL lite sampling"):
            ts = self.scheduler_preprocess_input[1]
            ts.set_data_from_numpy(np.array(i).astype(np.int32))
            scheduler_preprocess_output = self.scheduler_preprocess.predict([x, ts, s_in])
            noised_input, sigma_hat_s, next_sigma, sigma_hat = (
                scheduler_preprocess_output[0],
                scheduler_preprocess_output[1],
                scheduler_preprocess_output[2],
                scheduler_preprocess_output[3],
            )
            denoiser_output = self.denoiser.predict([sigma_hat_s])
            c_skip, c_out, c_in, c_noise = (
                denoiser_output[0],
                denoiser_output[1],
                denoiser_output[2],
                denoiser_output[3],
            )

            predict_noise_noised_input = self.predict_noise_input[0]
            predict_noise_c_noise_input = self.predict_noise_input[1]
            predict_noise_noised_input.set_data_from_numpy(noised_input.get_data_to_numpy() * c_in.get_data_to_numpy())
            predict_noise_c_noise_input.set_data_from_numpy(c_noise.get_data_to_numpy())
            model_output = self.predict_noise.predict(
                [predict_noise_noised_input, predict_noise_c_noise_input, context, y]
            )[0]

            noisy_sample_model_output_input = self.noisy_sample_input[0]
            noisy_sample_c_out_input = self.noisy_sample_input[1]
            noisy_sample_noised_input_input = self.noisy_sample_input[2]
            noisy_sample_c_skip_input = self.noisy_sample_input[3]
            noisy_sample_scale_input = self.noisy_sample_input[4]
            noisy_sample_x_input = self.noisy_sample_input[5]
            noisy_sample_sigma_hat_input = self.noisy_sample_input[6]
            noisy_sample_next_sigma_input = self.noisy_sample_input[7]

            noisy_sample_scale_input.set_data_from_numpy(scale)
            noisy_sample_model_output_input.set_data_from_numpy(model_output.get_data_to_numpy().astype(np.float32))
            noisy_sample_c_out_input.set_data_from_numpy(c_out.get_data_to_numpy().astype(np.float32))
            noisy_sample_noised_input_input.set_data_from_numpy(noised_input.get_data_to_numpy())
            noisy_sample_c_skip_input.set_data_from_numpy(c_skip.get_data_to_numpy().astype(np.float32))
            noisy_sample_x_input.set_data_from_numpy(x.get_data_to_numpy().astype(np.float32))
            noisy_sample_sigma_hat_input.set_data_from_numpy(sigma_hat.get_data_to_numpy().astype(np.float32))
            noisy_sample_next_sigma_input.set_data_from_numpy(next_sigma.get_data_to_numpy().astype(np.float32))

            x = self.noisy_sample.predict(
                [
                    noisy_sample_model_output_input,
                    noisy_sample_c_out_input,
                    noisy_sample_noised_input_input,
                    noisy_sample_c_skip_input,
                    noisy_sample_scale_input,
                    noisy_sample_x_input,
                    noisy_sample_sigma_hat_input,
                    noisy_sample_next_sigma_input,
                ]
            )[0]
        vae_decoder_input = self.vae_decoder_input[0]
        vae_decoder_input.set_data_from_numpy(x.get_data_to_numpy())
        image = self.vae_decoder.predict([vae_decoder_input])[0]
        image = image.get_data_to_numpy()
        return image


class SDLiteText2Img(SDLite):
    def __init__(
        self,
        data_prepare,
        scheduler_preprocess,
        predict_noise,
        noisy_sample,
        vae_decoder,
        denoiser,
        scheduler_prepare_sampling_loop,
        device_target="ascend",
        device_id=0,
        num_inference_steps=50,
    ):
        super(SDLiteText2Img, self).__init__(
            data_prepare,
            scheduler_preprocess,
            predict_noise,
            noisy_sample,
            vae_decoder,
            denoiser,
            scheduler_prepare_sampling_loop,
            device_target,
            device_id,
            num_inference_steps,
        )

    def data_prepare_predict(self, inputs):
        self.data_prepare_input[0].set_data_from_numpy(inputs["pos_clip_token"])
        self.data_prepare_input[1].set_data_from_numpy(inputs["pos_time_token"])
        self.data_prepare_input[2].set_data_from_numpy(inputs["neg_clip_token"])
        self.data_prepare_input[3].set_data_from_numpy(inputs["neg_time_token"])
        self.data_prepare_input[4].set_data_from_numpy(inputs["noise"])
        return self.data_prepare.predict(self.data_prepare_input)
