import logging

import mindspore as ms
from mindspore import nn, ops

from ..diffusion import SpacedDiffusion
from ..diffusion.diffusion_utils import _extract_into_tensor, discretized_gaussian_log_likelihood, mean_flat, normal_kl

__all__ = ["DiffusionWithLoss"]

logger = logging.getLogger(__name__)


class DiffusionWithLoss(nn.Cell):
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
        vae: nn.Cell = None,
        scale_factor: float = 0.18215,
        condition: str = "class",
        text_encoder: nn.Cell = None,
        cond_stage_trainable: bool = False,
        text_emb_cached: bool = True,
        video_emb_cached: bool = False,
    ):
        super().__init__()
        # TODO: is set_grad() necessary?
        self.network = network.set_grad()
        self.vae = vae
        self.diffusion = diffusion
        if condition is not None:
            assert isinstance(condition, str)
            condition = condition.lower()
        self.condition = condition
        self.text_encoder = text_encoder

        self.scale_factor = scale_factor
        self.cond_stage_trainable = cond_stage_trainable

        self.text_emb_cached = text_emb_cached
        self.video_emb_cached = video_emb_cached

        if self.text_emb_cached:
            self.text_encoder = None
            logger.info("Train with text embedding inputs")
        else:
            raise NotImplementedError
        if self.video_emb_cached:
            raise NotImplementedError

        if self.cond_stage_trainable and self.text_encoder:
            self.text_encoder.set_train(True)
            self.text_encoder.set_grad(True)

    def get_condition_embeddings(self, text_tokens, **kwargs):
        # text conditions inputs for cross-attention
        # optional: for some conditions, concat to latents, or add to time embedding
        if self.cond_stage_trainable:
            text_emb = self.text_encoder(text_tokens, **kwargs)
        else:
            text_emb = ops.stop_gradient(self.text_encoder(text_tokens, **kwargs))

        return text_emb

    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        image_latents = image_latents * self.scale_factor
        return image_latents.astype(ms.float16)

    def vae_decode(self, x):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        b, c, h, w = x.shape

        y = self.vae.decode(x / self.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b 3 H W) -> (b H W 3)
        y = ops.transpose(y, (0, 2, 3, 1))

        return y

    def vae_decode_video(self, x):
        """
        Args:
            x: (b f c h w), denoised latent
        Return:
            y: (b f H W 3), batch of images, normalized to [0, 1]
        """
        b, f, c, h, w = x.shape
        y = []
        for x_sample in x:
            y.append(self.vae_decode(x_sample))

        y = ops.stack(y, axis=0)

        return y

    def get_latents(self, x):
        if x.dim() == 5:
            # "b f c h w -> (b f) c h w"
            B, F, C, H, W = x.shape
            if C != 3:
                raise ValueError("Expect input shape (b f 3 h w), but get {}".format(x.shape))
            x = ops.reshape(x, (-1, C, H, W))

            z = ops.stop_gradient(self.vae_encode(x))

            # (b*f c h w) -> (b f c h w)
            z = ops.reshape(z, (B, F, z.shape[1], z.shape[2], z.shape[3]))

            # (b f c h w) -> (b c f h w)
            z = ops.transpose(z, (0, 2, 1, 3, 4))
        elif x.dim() == 4:
            B, C, H, W = x.shape
            if C != 3:
                raise ValueError("Expect input shape (b f 3 h w), but get {}".format(x.shape))
            z = ops.stop_gradient(self.vae_encode(x))
        else:
            raise ValueError("Incorrect Dimensions of x")
        return z

    def construct(self, x: ms.Tensor, text_tokens: ms.Tensor, mask: ms.Tensor = None):
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
        # 1. get image/video latents z using vae
        x = self.get_latents(x)

        # 2. get conditions
        if not self.text_emb_cached:
            text_embed = self.get_condition_embeddings(text_tokens)
        else:
            text_embed = text_tokens  # dataset retunrs text embeddings instead of text tokens

        loss = self.compute_loss(x, text_embed, mask)

        return loss

    def apply_model(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def _cal_vb(self, model_output, model_var_values, x, x_t, t):
        # make sure all inputs are fp32 for accuracy
        model_output = model_output.to(ms.float32)
        model_var_values = model_var_values.to(ms.float32)
        x = x.to(ms.float32)
        x_t = x_t.to(ms.float32)

        true_mean, _, true_log_variance_clipped = self.diffusion.q_posterior_mean_variance(x_start=x, x_t=x_t, t=t)
        # p_mean_variance(model=lambda *_: frozen_out, x_t, t, clip_denoised=False) begin
        min_log = _extract_into_tensor(self.diffusion.posterior_log_variance_clipped, t, x_t.shape)
        max_log = _extract_into_tensor(self.diffusion.log_betas, t, x_t.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        pred_xstart = self.diffusion.predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
        model_mean, _, _ = self.diffusion.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
        # assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        # p_mean_variance end
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = mean_flat(kl) / ms.numpy.log(2.0)  # TODO: 

        # print('D--: kl input type ', t.dtype, x.dtype,  model_mean.dtype, kl.dtype)

        # NOTE: make sure it's computed in fp32 since this func contains many exp.
        decoder_nll = -discretized_gaussian_log_likelihood(x, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = mean_flat(decoder_nll) / ms.numpy.log(2.0)

        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        vb = ops.where((t == 0), decoder_nll.to(kl.dtype), kl)

        return vb

    def compute_loss(self, x, text_embed, mask):
        t = ops.randint(0, self.diffusion.num_timesteps, (x.shape[0],))
        noise = ops.randn_like(x)
        x_t = self.diffusion.q_sample(x, t, noise=noise)

        # latte forward input match
        # text embed: (b n_tokens  d) -> (b  1 n_tokens d)
        text_embed = ops.expand_dims(text_embed, axis=1)
        model_output = self.apply_model(x_t, t, text_embed, mask)

        # (b c t h w),
        B, C, F = x_t.shape[:3]
        assert model_output.shape == (B, C * 2, F) + x_t.shape[3:]
        model_output, model_var_values = ops.split(model_output, C, axis=1)
        
        # Learn the variance using the variational bound, but don't let it affect our mean prediction.
        vb = self._cal_vb(ops.stop_gradient(model_output), model_var_values, x, x_t, t)

        loss = mean_flat((noise - model_output) ** 2) + vb
        loss = loss.mean()
        return loss
