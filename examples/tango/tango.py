import mindspore as ms
import mindspore.nn as nn
from tqdm import tqdm
from models import AudioDiffusion  # , DDPMScheduler
from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL


def init_from_ckpt(model, paths, ignore_keys=list()):
    for path in paths:
        sd = ms.load_checkpoint(path)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        ms.load_param_into_net(model, sd, strict_load=False)
        print(f"Restored from {path}")


class Tango(nn.Cell):
    def __init__(
        self,
        vae_config,
        stft_config,
        main_config,
    ):
        super().__init__()

        # vae_config = json.load(open("{}/vae_config.json".format(path)))
        # stft_config = json.load(open("{}/stft_config.json".format(path)))
        # main_config = json.load(open("{}/main_config.json".format(path)))

        self.vae = AutoencoderKL(**vae_config)
        self.stft = TacotronSTFT(**stft_config)
        self.model = AudioDiffusion(**main_config)

        # vae_weights = torch.load("{}/pytorch_model_vae.bin".format(path), map_location=device)
        # stft_weights = torch.load("{}/pytorch_model_stft.bin".format(path), map_location=device)
        # main_weights = torch.load("{}/pytorch_model_main.bin".format(path), map_location=device)

        # self.vae.load_state_dict(vae_weights)
        # self.stft.load_state_dict(stft_weights)
        # self.model.load_state_dict(main_weights)
        # init_from_ckpt(self, [vae_weights, stft_weights, main_weights])

        # self.vae.eval()
        # self.stft.eval()
        # self.model.eval()

        # self.scheduler = DDPMScheduler.from_pretrained(main_config["scheduler_name"], subfolder="scheduler")

    def chunks(self, lst, n):
        """ Yield successive n-sized chunks from a list. """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def generate(self, prompt, steps=100, guidance=3, samples=1, disable_progress=True):
        """ Genrate audio for a single prompt string. """

        latents = self.model.inference([prompt], self.scheduler, steps, guidance, samples, disable_progress=disable_progress)
        mel = self.vae.decode_first_stage(latents)
        wave = self.vae.decode_to_waveform(mel)
        return wave[0]

    def generate_for_batch(self, prompts, steps=100, guidance=3, samples=1, batch_size=8, disable_progress=True):
        """ Genrate audio for a list of prompt strings. """
        outputs = []
        for k in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[k: k+batch_size]
            latents = self.model.inference(batch, self.scheduler, steps, guidance, samples, disable_progress=disable_progress)
            mel = self.vae.decode_first_stage(latents)
            wave = self.vae.decode_to_waveform(mel)
            outputs += [item for item in wave]
        if samples == 1:
            return outputs
        else:
            return list(self.chunks(outputs, samples))
