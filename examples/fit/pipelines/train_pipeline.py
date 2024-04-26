from typing import Any, Dict, Optional

from diffusion import SpacedDiffusion
from diffusion.diffusion_utils import discretized_gaussian_log_likelihood, extract_into_tensor, mean_flat, normal_kl

import mindspore as ms
from mindspore import Tensor, nn, ops


class NetworkWithLoss(nn.Cell):
    def __init__(
        self,
        network: nn.Cell,
        diffusion: SpacedDiffusion,
        vae: Optional[nn.Cell] = None,
        scale_factor: float = 0.18215,
        condition: Optional[str] = "class",
        model_config: Dict[str, Any] = {},
    ):
        super().__init__(auto_prefix=False)
        self.network = network.set_grad()
        self.vae = vae
        self.diffusion = diffusion
        self.model_config = model_config

        if self.vae is None:
            self.latent_input = True
        else:
            self.latent_input = False

        if condition is not None:
            assert isinstance(condition, str)
            condition = condition.lower()

        self.condition = condition

        self.scale_factor = scale_factor

    def vae_encode(self, x):
        if self.latent_input:
            image_latents = x
        else:
            image_latents = ops.stop_gradient(self.vae.encode(x))
        image_latents = image_latents * self.scale_factor
        return image_latents.astype(ms.float16)

    def get_latents(self, x):
        return self.vae_encode(x)

    def _cal_vb(self, model_output, model_var_values, x, x_t, t, mask=None):
        true_mean, _, true_log_variance_clipped = self.diffusion.q_posterior_mean_variance(x_start=x, x_t=x_t, t=t)

        min_log = extract_into_tensor(self.diffusion.posterior_log_variance_clipped, t, x_t.shape)
        max_log = extract_into_tensor(ops.log(self.diffusion.betas), t, x_t.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        pred_xstart = self.diffusion.predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
        model_mean, _, _ = self.diffusion.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = mean_flat(kl, mask=mask) / ms.numpy.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(x, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = mean_flat(decoder_nll, mask=mask) / ms.numpy.log(2.0)
        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        vb = ops.where((t == 0), decoder_nll, kl)
        return vb

    def construct(
        self,
        x: ms.Tensor,
        labels: ms.Tensor,
        pos: ms.Tensor,
        mask: ms.Tensor,
    ):
        """
        Diffusion model forward and loss computation for training

        Args:
            x: flattened latent, in shape [bs, T, patch_size * patch_size * C]
            labels: class label ids [bs, ], optional
            text: text tokens padded to fixed shape [bs, 77], optional

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume input/output shape: (b c h w)
        """
        # 1. get scaled latent
        x = self.get_latents(x)

        # 2. get conditions
        if self.condition == "class":
            y = labels
        else:
            y = None

        loss = self.compute_loss(x, y, pos, mask)
        return loss

    def apply_model(self, x_t, t, y, pos, mask):
        return self.network(x_t, t, y=y, pos=pos, mask=mask)

    def unpatchify(self, x):
        p = self.model_config["patch_size"]
        c = self.model_config["C"]
        nh = self.model_config["H"] // p
        nw = self.model_config["W"] // p
        x = ops.reshape(x, (x.shape[0], nh, nw, p, p, c))
        x = ops.transpose(x, (0, 5, 1, 3, 2, 4))
        x = ops.reshape(x, (x.shape[0], c, nh * p, nw * p))
        return x

    def compute_loss(self, x: Tensor, y: Tensor, pos: Tensor, mask: Tensor) -> Tensor:
        D = x.shape[2]
        # convert x to 4-dim first for q_sample, prevent potential bug
        x = self.unpatchify(x)
        t = ops.randint(0, self.diffusion.num_timesteps, (x.shape[0],))

        noise = ops.randn_like(x)
        x_t = self.diffusion.q_sample(x, t, noise=noise)
        model_output = self.apply_model(x_t, t, y, pos, mask)

        B, C = x_t.shape[:2]
        assert model_output.shape == (B, C * 2) + x_t.shape[2:]
        model_output, model_var_values = ops.split(model_output, C, axis=1)

        # Learn the variance using the variational bound, but don't let it affect our mean prediction.
        mask = self.unpatchify(ops.tile(mask[..., None], (1, 1, D)))
        vb = self._cal_vb(ops.stop_gradient(model_output), model_var_values, x, x_t, t, mask=mask)

        loss = mean_flat((noise - model_output) ** 2, mask=mask) + vb
        loss = loss.mean()
        return loss
