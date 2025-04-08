# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any, Dict

from sparktts.modules.encoder_decoder.feat_decoder import Decoder
from sparktts.modules.encoder_decoder.feat_encoder import Encoder
from sparktts.modules.encoder_decoder.wave_generator import WaveGenerator
from sparktts.modules.speaker.speaker_encoder import SpeakerEncoder
from sparktts.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize
from sparktts.utils.file import load_config

import mindspore as ms
from mindspore import mint, nn

from mindone.diffusers.models.model_loading_utils import load_state_dict
from mindone.diffusers.models.modeling_utils import _convert_state_dict


class BiCodec(nn.Cell):
    """
    BiCodec model for speech synthesis, incorporating a speaker encoder, feature encoder/decoder,
    quantizer, and wave generator.
    """

    def __init__(
        self,
        mel_params: Dict[str, Any],
        encoder: nn.Cell,
        decoder: nn.Cell,
        quantizer: nn.Cell,
        speaker_encoder: nn.Cell,
        prenet: nn.Cell,
        postnet: nn.Cell,
        **kwargs,
    ) -> None:
        """
        Initializes the BiCodec model with the required components.

        Args:
            mel_params (dict): Parameters for the mel-spectrogram transformer.
            encoder (nn.Cell): Encoder module.
            decoder (nn.Cell): Decoder module.
            quantizer (nn.Cell): Quantizer module.
            speaker_encoder (nn.Cell): Speaker encoder module.
            prenet (nn.Cell): Prenet network.
            postnet (nn.Cell): Postnet network.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.speaker_encoder = speaker_encoder
        self.prenet = prenet
        self.postnet = postnet
        self.init_mel_transformer(mel_params)

    @classmethod
    def load_from_checkpoint(cls, model_dir: Path, **kwargs) -> "BiCodec":
        """
        Loads the model from a checkpoint.

        Args:
            model_dir (Path): Path to the model directory containing checkpoint and config.

        Returns:
            BiCodec: The initialized BiCodec model.
        """
        ckpt_path = f"{model_dir}/model.safetensors"
        config = load_config(f"{model_dir}/config.yaml")["audio_tokenizer"]
        mel_params = config["mel_params"]
        encoder = Encoder(**config["encoder"])
        quantizer = FactorizedVectorQuantize(**config["quantizer"])
        prenet = Decoder(**config["prenet"])
        postnet = Decoder(**config["postnet"])
        decoder = WaveGenerator(**config["decoder"])
        speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])

        model = cls(
            mel_params=mel_params,
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            speaker_encoder=speaker_encoder,
            prenet=prenet,
            postnet=postnet,
        )

        state_dict = load_state_dict(ckpt_path)
        state_dict = _convert_state_dict(model, state_dict)
        missing_keys, unexpected_keys = ms.load_param_into_net(model, state_dict)

        for key in missing_keys:
            print(f"param_not_load: {key}")
        for key in unexpected_keys:
            print(f"ckpt_not_load: {key}")

        # model.eval()
        # model.remove_weight_norm()

        return model

    def construct(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a forward pass through the model.

        Args:
            batch (dict): A dictionary containing features, reference waveform, and target waveform.

        Returns:
            dict: A dictionary containing the reconstruction, features, and other metrics.
        """
        feat = batch["feat"]
        mel = self.mel_transformer(batch["ref_wav"]).squeeze(1)

        z = self.encoder(feat.transpose(1, 2))
        vq_outputs = self.quantizer(z)

        x_vector, d_vector = self.speaker_encoder(mel.transpose(1, 2))

        conditions = d_vector
        with_speaker_loss = False

        x = self.prenet(vq_outputs["z_q"], conditions)
        pred_feat = self.postnet(x)
        x = x + conditions.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return {
            "vq_loss": vq_outputs["vq_loss"],
            "perplexity": vq_outputs["perplexity"],
            "cluster_size": vq_outputs["active_num"],
            "recons": wav_recon,
            "pred_feat": pred_feat,
            "x_vector": x_vector,
            "d_vector": d_vector,
            "audios": batch["wav"].unsqueeze(1),
            "with_speaker_loss": with_speaker_loss,
        }

    def tokenize(self, batch: Dict[str, Any]):
        """
        Tokenizes the input audio into semantic and global tokens.

        Args:
            batch (dict): The input audio features and reference waveform.

        Returns:
            tuple: Semantic tokens and global tokens.
        """
        feat = batch["feat"]
        mel = ms.tensor(self.mel_transformer(batch["ref_wav"].numpy())).squeeze(1)

        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        global_tokens = self.speaker_encoder.tokenize(mel.transpose(1, 2))

        return semantic_tokens, global_tokens

    def detokenize(self, semantic_tokens, global_tokens):
        """
        Detokenizes the semantic and global tokens into a waveform.

        Args:
            semantic_tokens (tensor): Semantic tokens.
            global_tokens (tensor): Global tokens.

        Returns:
            tensor: Reconstructed waveform.
        """
        z_q = self.quantizer.detokenize(semantic_tokens)
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        x = self.prenet(z_q, d_vector)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return wav_recon

    def init_mel_transformer(self, config: Dict[str, Any]):
        """
        Initializes the MelSpectrogram transformer based on the provided configuration.

        Args:
            config (dict): Configuration parameters for MelSpectrogram.
        """
        import mindspore.dataset.audio as msaudio

        self.mel_transformer = msaudio.MelSpectrogram(
            config["sample_rate"],
            config["n_fft"],
            config["win_length"],
            config["hop_length"],
            float(config["mel_fmin"]),
            config["mel_fmax"],
            n_mels=config["num_mels"],
            power=1,
            norm=msaudio.NormType.SLANEY,
            mel_scale=msaudio.MelType.SLANEY,
        )


# Test the model
if __name__ == "__main__":
    config = load_config("pretrained_models/SparkTTS-0.5B/BiCodec/config.yaml")
    model = BiCodec.load_from_checkpoint(
        model_dir="pretrained_models/SparkTTS-0.5B/BiCodec",
    )

    # Generate random inputs for testing
    duration = 0.96
    x = mint.randn(20, 1, int(duration * 16000))
    feat = mint.randn(20, int(duration * 50), 1024)
    inputs = {"feat": feat, "wav": x, "ref_wav": x}

    # Forward pass
    outputs = model(inputs)
    semantic_tokens, global_tokens = model.tokenize(inputs)
    wav_recon = model.detokenize(semantic_tokens, global_tokens)

    # Verify if the reconstruction matches
    if mint.allclose(outputs["recons"].detach(), wav_recon):
        print("Test successful")
    else:
        print("Test failed")
