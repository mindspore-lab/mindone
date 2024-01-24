from typing import Optional, Tuple, Union

from gm.modules.embedders.modules import AbstractEmbModel
from gm.util import append_dims, instantiate_from_config

from mindspore import Tensor
from mindspore import dtype as ms_dtype
from mindspore import ops


class VideoPredictionEmbedderWithEncoder(AbstractEmbModel):
    def __init__(
        self,
        n_cond_frames: int,
        n_copies: int,
        encoder_config: dict,
        sigma_sampler_config: Optional[dict] = None,
        sigma_cond_config: Optional[dict] = None,
        is_ae: bool = False,
        scale_factor: float = 1.0,
        disable_encoder_amp: bool = False,
        en_and_decode_n_samples_a_time: int = 0,
    ):
        super().__init__()

        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.encoder = instantiate_from_config(encoder_config)
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config) if sigma_sampler_config is not None else None
        self.sigma_cond = instantiate_from_config(sigma_cond_config) if sigma_cond_config is not None else None
        self.is_ae = is_ae
        self.scale_factor = scale_factor
        self.disable_encoder_amp = disable_encoder_amp
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

        if disable_encoder_amp:
            self.encoder.to_float(ms_dtype.float32)

    def construct(
        self, vid: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, dict], Tuple[Tuple[Tensor, Tensor], dict]]:
        sigma_cond = None
        if self.sigma_sampler is not None:
            b = vid.shape[0] // self.n_cond_frames
            sigmas = self.sigma_sampler(b)
            if self.sigma_cond is not None:
                sigma_cond = self.sigma_cond(sigmas)
                sigma_cond = sigma_cond.repeat(self.n_copies, axis=0)  # b d -> (b t) d
            sigmas = sigmas.repeat(self.n_copies, axis=0)  # b -> (b t)
            noise = ops.randn_like(vid)
            vid = vid + noise * append_dims(sigmas, vid.ndim)

        if self.disable_encoder_amp:
            vid = vid.astype(ms_dtype.float32)

        n_samples = self.en_and_decode_n_samples_a_time or vid.shape[0]
        all_out = []
        for n in range(0, vid.shape[0], n_samples):
            if self.is_ae:
                out = self.encoder.encode(vid[n : n + n_samples])
            else:
                out = self.encoder(vid[n : n + n_samples])
            all_out.append(out)

        vid = ops.cat(all_out, axis=0)
        vid *= self.scale_factor

        vid = vid.reshape(-1, self.n_cond_frames, *vid.shape[1:])  # (b t) c h w -> b t c h w
        vid = vid.reshape(vid.shape[0], -1, *vid.shape[3:])  # b t c h w -> b (t c) h w
        vid = vid.repeat(self.n_copies, axis=0)  # b (t c) h w -> (b s) (t c) h w

        if self.sigma_cond is not None:
            return vid, sigma_cond
        return vid


class FrozenOpenCLIPImagePredictionEmbedder(AbstractEmbModel):
    def __init__(
        self,
        open_clip_embedding_config: dict,
        n_cond_frames: int,
        n_copies: int,
    ):
        super().__init__()

        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.open_clip = instantiate_from_config(open_clip_embedding_config)

    def construct(self, vid: Tensor) -> Tensor:
        vid = self.open_clip(vid)
        vid = vid.reshape(-1, self.n_cond_frames, vid.shape[1])  # (b t) d -> b t d
        vid = vid.repeat(self.n_copies, axis=0)  # b t d -> (b s) t d

        return vid
