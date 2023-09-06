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
        n_infer_steps = mslite.Tensor()
        n_infer_steps.shape = []
        n_infer_steps.dtype = mslite.DataType.INT32
        n_infer_steps.set_data_from_numpy(np.array(num_inference_steps, np.int32))
        self.num_inference_steps = n_infer_steps

        # get input
        self.data_prepare_input = self.data_prepare.get_inputs()
        self.predict_noise_input = self.predict_noise.get_inputs()
        self.noisy_sample_input = self.noisy_sample.get_inputs()
        self.vae_decoder_input = self.vae_decoder.get_inputs()

    @abstractmethod
    def data_prepare_predict(self, inputs):
        pass

    def __call__(self, inputs):
        predict_outputs = self.data_prepare_predict(inputs)
        if len(predict_outputs) == 2:
            c_crossattn, latents = predict_outputs
        elif len(predict_outputs) == 3:
            c_crossattn, latents, c_concat = predict_outputs
        else:
            raise ValueError("data_prepare_predict error")
        scale = self.predict_noise_input[3]
        scale.set_data_from_numpy(np.array(inputs["scale"]))
        iterator = tqdm(inputs["timesteps"], desc="Stable Diffusion Sampling", total=len(inputs["timesteps"]))
        for i, t in enumerate(iterator):
            # predict the noise residual
            ts = self.predict_noise_input[1]
            ts.set_data_from_numpy(np.array(t).astype(np.int32))
            latents = self.scheduler_preprocess.predict([latents, ts])[0]
            if len(predict_outputs) == 2:
                noise_pred = self.predict_noise.predict([latents, ts, c_crossattn, scale])[0]
            else:
                noise_pred = self.predict_noise.predict([latents, ts, c_crossattn, scale, c_concat])[0]
            latents = self.noisy_sample.predict([noise_pred, ts, latents, self.num_inference_steps])[0]
        image = self.vae_decoder.predict([latents])[0]
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
            device_target,
            device_id,
            num_inference_steps,
        )

    def data_prepare_predict(self, inputs):
        self.data_prepare_input[0].set_data_from_numpy(inputs["prompt_data"])
        self.data_prepare_input[1].set_data_from_numpy(inputs["negative_prompt_data"])
        self.data_prepare_input[2].set_data_from_numpy(inputs["noise"])
        return self.data_prepare.predict(self.data_prepare_input)


class SDLiteImg2Img(SDLite):
    def __init__(
        self,
        data_prepare,
        scheduler_preprocess,
        predict_noise,
        noisy_sample,
        vae_decoder,
        device_target="ascend",
        device_id=0,
        num_inference_steps=50,
    ):
        super(SDLiteImg2Img, self).__init__(
            data_prepare,
            scheduler_preprocess,
            predict_noise,
            noisy_sample,
            vae_decoder,
            device_target,
            device_id,
            num_inference_steps,
        )

    def data_prepare_predict(self, inputs):
        t0 = self.data_prepare_input[4]
        timesteps = inputs["timesteps"]
        t0.set_data_from_numpy(np.array(timesteps[0]).astype(np.int32))
        self.data_prepare_input[0].set_data_from_numpy(inputs["prompt_data"])
        self.data_prepare_input[1].set_data_from_numpy(inputs["negative_prompt_data"])
        self.data_prepare_input[2].set_data_from_numpy(inputs["img"])
        self.data_prepare_input[3].set_data_from_numpy(inputs["noise"])
        self.data_prepare_input[4].set_data_from_numpy(t0)
        return self.data_prepare.predict(self.data_prepare_input)


class SDLiteInpaint(SDLite):
    def __init__(
        self,
        data_prepare,
        scheduler_preprocess,
        predict_noise,
        noisy_sample,
        vae_decoder,
        device_target="ascend",
        device_id=0,
        num_inference_steps=50,
    ):
        super(SDLiteInpaint, self).__init__(
            data_prepare,
            scheduler_preprocess,
            predict_noise,
            noisy_sample,
            vae_decoder,
            device_target,
            device_id,
            num_inference_steps,
        )

    def data_prepare_predict(self, inputs):
        self.data_prepare_input[0].set_data_from_numpy(inputs["prompt_data"])
        self.data_prepare_input[1].set_data_from_numpy(inputs["negative_prompt_data"])
        self.data_prepare_input[2].set_data_from_numpy(inputs["masked_image"])
        self.data_prepare_input[3].set_data_from_numpy(inputs["mask"])
        self.data_prepare_input[4].set_data_from_numpy(inputs["noise"])
        return self.data_prepare.predict(self.data_prepare_input)
