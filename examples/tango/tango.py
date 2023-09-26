import json
import os

from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from models import AudioDiffusion
from tqdm import tqdm

import mindspore.nn as nn


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Tango(nn.Cell):
    def __init__(
        self,
        config_path,
    ):
        super().__init__()

        main_config = dotdict(json.load(open(os.path.join(config_path, "config.json"))))
        vae_config = dotdict(json.load(open(main_config.pop("vae_model_config_path"))))
        stft_config = dotdict(json.load(open(main_config.pop("stft_model_config_path"))))

        self.vae = AutoencoderKL(**vae_config)
        self.stft = TacotronSTFT(**stft_config)
        self.model = AudioDiffusion(**main_config)

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from a list."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def generate(self, prompt, sampler, steps=100, guidance=3, samples=1, disable_progress=True):
        """Genrate audio for a single prompt string."""
        latents = self.model.inference(
            [prompt],
            steps,
            guidance,
            samples,
            disable_progress,
            sampler=sampler,
            padding=False,
            truncation=False,
        )

        mel = self.vae.decode_first_stage(latents)
        wave = self.vae.decode_to_waveform(mel)
        return wave[0]

    def generate_for_batch(self, prompts, steps=100, guidance=3, samples=1, batch_size=8, disable_progress=True):
        """Genrate audio for a list of prompt strings."""
        outputs = []
        for k in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[k : k + batch_size]
            latents = self.model.inference(
                batch, self.scheduler, steps, guidance, samples, disable_progress=disable_progress
            )
            mel = self.vae.decode_first_stage(latents)
            wave = self.vae.decode_to_waveform(mel)
            outputs += [item for item in wave]
        if samples == 1:
            return outputs
        else:
            return list(self.chunks(outputs, samples))
