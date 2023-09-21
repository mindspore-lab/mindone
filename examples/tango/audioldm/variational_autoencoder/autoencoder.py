from audioldm.hifigan.utilities import get_vocoder, vocoder_infer
from audioldm.variational_autoencoder.modules import Decoder, Encoder
from einops import rearrange

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class AutoencoderKL(nn.Cell):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        image_key="fbank",
        embed_dim=None,
        time_shuffle=1,
        subband=1,
        ckpt_path=None,
        reload_from_ckpt=None,
        ignore_keys=[],
        colorize_nlabels=None,
        monitor=None,
        base_learning_rate=1e-5,
        scale_factor=1,
        use_fp16=False,
    ):
        super().__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        self.quant_conv = nn.Conv2d(
            2 * ddconfig["z_channels"], 2 * embed_dim, 1, pad_mode="valid", has_bias=True
        ).to_float(self.dtype)
        self.post_quant_conv = nn.Conv2d(
            embed_dim, ddconfig["z_channels"], 1, pad_mode="valid", has_bias=True
        ).to_float(self.dtype)

        self.vocoder = get_vocoder(config=None, trainable=False)
        self.embed_dim = embed_dim

        if monitor is not None:
            self.monitor = monitor

        self.time_shuffle = time_shuffle
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None

        self.scale_factor = scale_factor

    def encode(self, x):
        # x = self.time_shuffle_operation(x)
        x = self.freq_split_subband(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = self.split(moments)
        logvar = ops.clip_by_value(logvar, -30.0, 20.0)
        std = self.exp(0.5 * logvar)
        x = mean + std * self.stdnormal(mean.shape)
        return x

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec = self.freq_merge_subband(dec)
        return dec

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = ms.load_checkpoint(path)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        ms.load_param_into_net(self, sd, strict_load=False)
        print(f"Restored from {path}")

    def decode_to_waveform(self, dec):
        dec = dec.squeeze(1).permute(0, 2, 1)
        wav_reconstruction = vocoder_infer(dec, self.vocoder)
        return wav_reconstruction

    def construct(self, input, sample_posterior=True):
        z = self.encode(input)
        dec = self.decode(z)
        return dec

    def freq_split_subband(self, fbank):
        if self.subband == 1 or self.image_key != "stft":
            return fbank

        bs, ch, tstep, fbins = fbank.size()

        assert fbank.size()[-1] % self.subband == 0
        assert ch == 1

        return fbank.squeeze(1).reshape(bs, tstep, self.subband, fbins // self.subband).permute(0, 2, 1, 3)

    def freq_merge_subband(self, subband_fbank):
        if self.subband == 1 or self.image_key != "stft":
            return subband_fbank
        assert subband_fbank.size(1) == self.subband  # Channel dimension
        bs, sub_ch, tstep, fbins = subband_fbank.size()
        return subband_fbank.permute(0, 2, 1, 3).reshape(bs, tstep, -1).unsqueeze(1)

    def encode_first_stage(self, x):
        return ops.stop_gradient(self.encode(x))

    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = ops.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z
        return ops.stop_gradient(self.decode(z))

    def get_first_stage_encoding(self, encoder_posterior):
        # z = encoder_posterior.sample()
        z = encoder_posterior
        return self.scale_factor * z
