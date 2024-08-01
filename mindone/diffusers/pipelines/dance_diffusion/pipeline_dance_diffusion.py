# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import numpy as np

from ...utils import logging
from ...utils.mindspore_utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DanceDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for audio generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet1DModel`]):
            A `UNet1DModel` to denoise the encoded audio.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded audio latents. Can be one of
            [`IPNDMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 100,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        audio_length_in_s: Optional[float] = None,
        return_dict: bool = True,
    ) -> Union[AudioPipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of audio samples to generate.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher-quality audio sample at
                the expense of slower inference.
            generator (`np.random.Generator`, *optional*):
                A [`np.random.Generator`](https://numpy.org/doc/stable/reference/random/generator.html) to make
                generation deterministic.
            audio_length_in_s (`float`, *optional*, defaults to `self.unet.config.sample_size/self.unet.config.sample_rate`):
                The length of the generated audio sample in seconds.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.AudioPipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        from mindone.diffusers import DiffusionPipeline
        from scipy.io.wavfile import write

        model_id = "harmonai/maestro-150k"
        pipe = DiffusionPipeline.from_pretrained(model_id)

        audios = pipe(audio_length_in_s=4.0)[0]

        # To save locally
        for i, audio in enumerate(audios):
            write(f"maestro_test_{i}.wav", pipe.unet.config.sample_rate, audio.transpose())

        # To dislay in google colab
        import IPython.display as ipd

        for audio in audios:
            display(ipd.Audio(audio, rate=pipe.unet.config.sample_rate))
        ```

        Returns:
            [`~pipelines.AudioPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.AudioPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated audio.
        """

        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size / self.unet.config.sample_rate

        sample_size = audio_length_in_s * self.unet.config.sample_rate

        down_scale_factor = 2 ** len(self.unet.up_blocks)
        if sample_size < 3 * down_scale_factor:
            raise ValueError(
                f"{audio_length_in_s} is too small. Make sure it's bigger or equal to"
                f" {3 * down_scale_factor / self.unet.config.sample_rate}."
            )

        original_sample_size = int(sample_size)
        if sample_size % down_scale_factor != 0:
            sample_size = (
                (audio_length_in_s * self.unet.config.sample_rate) // down_scale_factor + 1
            ) * down_scale_factor
            logger.info(
                f"{audio_length_in_s} is increased to {sample_size / self.unet.config.sample_rate} so that it can be handled"
                f" by the model. It will be cut to {original_sample_size / self.unet.config.sample_rate} after the denoising"
                " process."
            )
        sample_size = int(sample_size)

        dtype = next(self.unet.get_parameters()).dtype
        shape = (batch_size, self.unet.config.in_channels, sample_size)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        audio = randn_tensor(shape, generator=generator, dtype=dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.timesteps = self.scheduler.timesteps.to(dtype)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(audio, t)[0]

            # 2. compute previous audio sample: x_t -> t_t-1
            audio = self.scheduler.step(model_output, t, audio)[0]

        audio = audio.clamp(-1, 1).float().numpy()

        audio = audio[:, :, :original_sample_size]

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)
