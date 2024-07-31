from typing import Optional

from pixart.diffusion import SpacedDiffusion
from pixart.diffusion.diffusion_utils import (
    _extract_into_tensor,
    discretized_gaussian_log_likelihood,
    mean_flat,
    normal_kl,
)
from pixart.modules.pixart import PixArt

import mindspore as ms
from mindspore import Tensor, nn, ops

from mindone.diffusers import AutoencoderKL
from mindone.transformers import T5EncoderModel


class NetworkWithLoss(nn.Cell):
    def __init__(
        self,
        network: PixArt,
        diffusion: SpacedDiffusion,
        vae: Optional[AutoencoderKL] = None,
        text_encoder: Optional[T5EncoderModel] = None,
        scale_factor: float = 0.18215,
    ) -> None:
        super().__init__(auto_prefix=False)
        self.network = network
        self.vae = vae
        self.text_encoder = text_encoder
        self.diffusion = diffusion
        self.scale_factor = scale_factor

    def get_latents(self, x: Tensor) -> Tensor:
        if self.vae is None:
            image_latents = x
        else:
            image_latents = ops.stop_gradient(self.vae.encode(x))
        image_latents = image_latents * self.scale_factor
        return image_latents

    def get_text_emb(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.text_encoder is None:
            return x
        text_emb = ops.stop_gradient(self.text_encoder(x, mask))
        return text_emb

    def _cal_vb(self, model_output, model_var_values, x, x_t, t):
        true_mean, _, true_log_variance_clipped = self.diffusion.q_posterior_mean_variance(x_start=x, x_t=x_t, t=t)

        min_log = _extract_into_tensor(self.diffusion.posterior_log_variance_clipped, t, x_t.shape)
        max_log = _extract_into_tensor(ops.log(self.diffusion.betas), t, x_t.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        pred_xstart = self.diffusion.predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
        model_mean, _, _ = self.diffusion.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = mean_flat(kl) / ms.numpy.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(x, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = mean_flat(decoder_nll) / ms.numpy.log(2.0)
        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        vb = ops.where((t == 0), decoder_nll, kl)
        return vb

    def construct(self, x: Tensor, text: Tensor, text_mask: Tensor) -> Tensor:
        """
        Args:
            x: Image (with vae provided) / VAE latent (without vae provided)
            text: Text Token (with text_encoder provided) / Text Embed (without text_encoder provided)
            text_mask: Text Mask
        """
        x = self.get_latents(x)
        text_emb = self.get_text_emb(text, text_mask)
        loss = self.compute_loss(x, text_emb, text_mask)
        return loss

    def apply_model(self, x: Tensor, t: Tensor, text_emb: Tensor, text_mask: Tensor):
        return self.network(x, t, text_emb, text_mask)

    def compute_loss(self, x: Tensor, text_emb: Tensor, text_mask: Tensor) -> Tensor:
        t = ops.randint(0, self.diffusion.num_timesteps, (x.shape[0],))

        noise = ops.randn_like(x)
        x_t = self.diffusion.q_sample(x, t, noise=noise)
        model_output = self.apply_model(x_t, t, text_emb, text_mask)

        B, C = x_t.shape[:2]
        assert model_output.shape == (B, C * 2) + x_t.shape[2:]
        model_output, model_var_values = ops.split(model_output, C, axis=1)

        # Learn the variance using the variational bound, but don't let it affect our mean prediction.
        vb = self._cal_vb(ops.stop_gradient(model_output), model_var_values, x, x_t, t)

        loss = mean_flat((noise - model_output) ** 2) + vb
        loss = loss.mean()
        return loss
