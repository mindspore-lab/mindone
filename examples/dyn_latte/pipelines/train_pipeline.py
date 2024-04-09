import logging
from typing import Any, Dict

from diffusion import SpacedDiffusion
from diffusion.diffusion_utils import discretized_gaussian_log_likelihood, extract_into_tensor, mean_flat, normal_kl

import mindspore as ms
from mindspore import Tensor, nn, ops

__all__ = ["NetworkWithLoss", "get_model_with_loss"]

logger = logging.getLogger(__name__)


class NetworkWithLoss(nn.Cell):
    """An training pipeline for diffusion model

    Args:
        model (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        diffusion: (object): A class for Gaussian Diffusion.
        scale_factor (float): scale_factor for vae.
        condition (str): The type of conditions of model in [None, 'text', 'class'].
            If it is None, model is a un-conditional video generator.
            If it is 'text', model accepts text embeddings (B, T, N) as conditions, and generates videos.
            If it is 'class', model accepts class labels (B, ) as conditions, and generates videos.
        text_encoder (nn.Cell): A text encoding model which accepts token ids and returns text embeddings in shape (T, D).
            T is the number of tokens, and D is the embedding dimension.
        cond_stage_trainable (bool): whether to train the text encoder.
        train_with_embed (bool): whether to train with embeddings (no need vae and text encoder to extract latent features and text embeddings)
    """

    def __init__(
        self,
        network: nn.Cell,
        diffusion: SpacedDiffusion,
        scale_factor: float = 0.18215,
        condition: str = "class",
        text_encoder: nn.Cell = None,
        cond_stage_trainable: bool = False,
        model_config: Dict[str, Any] = {},
    ):
        super().__init__()
        self.model_config = model_config
        self.network = network.set_grad()
        self.diffusion = diffusion
        if condition is not None:
            assert isinstance(condition, str)
            condition = condition.lower()
        self.condition = condition
        self.text_encoder = text_encoder
        if self.condition == "text":
            assert self.text_encoder is not None, "Expect to get text encoder"

        self.scale_factor = scale_factor
        self.cond_stage_trainable = cond_stage_trainable

        if self.cond_stage_trainable and self.text_encoder:
            self.text_encoder.set_train(True)
            self.text_encoder.set_grad(True)

    def get_condition_embeddings(self, text_tokens):
        # text conditions inputs for cross-attention
        # optional: for some conditions, concat to latents, or add to time embedding
        if self.cond_stage_trainable:
            text_emb = self.text_encoder(text_tokens)
        else:
            text_emb = ops.stop_gradient(self.text_encoder(text_tokens))

        return text_emb

    def get_latents(self, x: Tensor):
        x = x * self.scale_factor
        return x.astype(ms.float16)

    def construct(self, x: Tensor, pos: Tensor, mask_t: Tensor, mask_s: Tensor, labels: Tensor, text_tokens: Tensor):
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames, resized and normalized to shape [bs, F, 3, 256, 256]
            text_tokens: text tokens padded to fixed shape [bs, 77]
            labels: the class labels

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume model input/output shape: (b c f h w)
                unet2d input/output shape: (b c h w)
        """
        x = self.get_latents(x)

        if self.condition == "text":
            text_embed = self.get_condition_embeddings(text_tokens)
        else:
            text_embed = None

        if self.condition == "class":
            y = labels
        else:
            y = None
        loss = self.compute_loss(x, pos, mask_t, mask_s, y, text_embed)
        return loss

    def apply_model(self, *args, **kwargs):
        return self.network(*args, **kwargs)

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

    def unpatchify(self, x):
        n = self.model_config["N"]
        p = self.model_config["patch_size"]
        c = self.model_config["C"]
        nh = self.model_config["H"] // p
        nw = self.model_config["W"] // p
        x = ops.reshape(x, (n, -1, nh, nw, p, p, c))
        x = ops.transpose(x, (0, 1, 6, 2, 4, 3, 5))
        x = ops.reshape(x, (n, -1, c, nh * p, nw * p))
        return x

    def compute_loss(self, x, pos, mask_t, mask_s, y, text_embed):
        D = x.shape[3]
        # convert x to 5-dim first for q_sample, prevent potential bug
        x = self.unpatchify(x)
        t = ops.randint(0, self.diffusion.num_timesteps, (x.shape[0],))

        noise = ops.randn_like(x)
        x_t = self.diffusion.q_sample(x, t, noise=noise)
        model_output = self.apply_model(x_t, t, pos, mask_t, mask_s, y=y, text_embed=text_embed)

        B, F, C = x_t.shape[:3]
        assert model_output.shape == (B, F, C * 2) + x_t.shape[3:]
        model_output, model_var_values = ops.split(model_output, C, axis=2)

        # Learn the variance using the variational bound, but don't let it affect our mean prediction.
        mask = ops.logical_and(mask_t[:, :, None], mask_s[:, None, :])
        mask = ops.reshape(mask, (mask.shape[0], -1, 1))
        mask = self.unpatchify(ops.tile(mask, (1, 1, D)))
        vb = self._cal_vb(ops.stop_gradient(model_output), model_var_values, x, x_t, t, mask=mask)

        loss = mean_flat((noise - model_output) ** 2, mask=mask) + vb
        loss = loss.mean()
        return loss


class UnconditionalModelWithLoss(NetworkWithLoss):
    def construct(self, x: Tensor, pos: Tensor, mask_s: Tensor, mask_t: Tensor):
        return super().construct(x, pos, mask_s, mask_t, labels=None, text_tokens=None)


class ClassConditionedModelWithLoss(NetworkWithLoss):
    def construct(self, x: Tensor, pos: Tensor, mask_s: Tensor, mask_t: Tensor, labels: Tensor):
        return super().construct(x, pos, mask_s, mask_t, labels=labels, text_tokens=None)


class TextConditionedModelWithLoss(NetworkWithLoss):
    def construct(self, x: Tensor, pos: Tensor, mask_s: Tensor, mask_t: Tensor, text_tokens: Tensor):
        return super().construct(x, pos, mask_s, mask_t, labels=None, text_tokens=text_tokens)


def get_model_with_loss(condition):
    if condition is None:
        return UnconditionalModelWithLoss
    elif condition == "class":
        return ClassConditionedModelWithLoss
    elif condition == "text":
        return TextConditionedModelWithLoss
    else:
        raise NotImplementedError
