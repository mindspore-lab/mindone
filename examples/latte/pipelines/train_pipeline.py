from diffusion import SpacedDiffusion
from diffusion.diffusion_utils import _extract_into_tensor, discretized_gaussian_log_likelihood, mean_flat, normal_kl

import mindspore as ms
from mindspore import nn, ops

__all__ = ["NetworkWithLoss", "get_model_with_loss"]


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
    """

    def __init__(
        self,
        network: nn.Cell,
        vae: nn.Cell,
        diffusion: SpacedDiffusion,
        scale_factor: float = 0.18215,
        condition: str = "class",
        text_encoder: nn.Cell = None,
        cond_stage_trainable: bool = False,
    ):
        super().__init__()
        self.network = network.set_grad()
        self.vae = vae
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

        if self.cond_stage_trainable:
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
        x = x.reshape((b * f, c, h, w))

        y = self.vae_decode(x)
        _, h, w, c = y.shape
        y = y.reshape((b, f, h, w, c))

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
        elif x.dim() == 4:
            B, C, H, W = x.shape
            if C != 3:
                raise ValueError("Expect input shape (b f 3 h w), but get {}".format(x.shape))
            z = ops.stop_gradient(self.vae_encode(x))
        else:
            raise ValueError("Incorrect Dimensions of x")
        return z

    def construct(self, x: ms.Tensor, labels: ms.Tensor, text_tokens: ms.Tensor, **kwargs):
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
        if self.condition == "text":
            text_embed = self.get_condition_embeddings(text_tokens)
        else:
            text_embed = None

        if self.condition == "class":
            y = labels
        else:
            y = None
        loss = self.compute_loss(x, y, text_embed)
        return loss

    def apply_model(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def _cal_vb(self, model_output, model_var_values, x, x_t, t):
        true_mean, _, true_log_variance_clipped = self.diffusion.q_posterior_mean_variance(x_start=x, x_t=x_t, t=t)
        # p_mean_variance(model=lambda *_: frozen_out, x_t, t, clip_denoised=False) begin
        min_log = _extract_into_tensor(self.diffusion.posterior_log_variance_clipped, t, x_t.shape)
        max_log = _extract_into_tensor(ops.log(self.diffusion.betas), t, x_t.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        pred_xstart = self.diffusion.predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
        model_mean, _, _ = self.diffusion.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
        # assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        # p_mean_variance end
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = mean_flat(kl) / ms.numpy.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(x, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = mean_flat(decoder_nll) / ms.numpy.log(2.0)
        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        vb = ops.where((t == 0), decoder_nll.to(kl.dtype), kl)
        return vb

    def compute_loss(self, x, y, text_embed):
        t = ops.randint(0, self.diffusion.num_timesteps, (x.shape[0],))
        noise = ops.randn_like(x)
        x_t = self.diffusion.q_sample(x, t, noise=noise)
        model_output = self.apply_model(x_t, t, y=y, text_embed=text_embed)
        if x_t.dim() == 5:
            B, F, C = x_t.shape[:3]
            assert model_output.shape == (B, F, C * 2) + x_t.shape[3:]
            model_output, model_var_values = ops.split(model_output, C, axis=2)
        else:
            B, C = x_t.shape[:2]
            assert model_output.shape == (B, C * 2) + x_t.shape[2:]
            model_output, model_var_values = ops.split(model_output, C, axis=1)

        # Learn the variance using the variational bound, but don't let it affect our mean prediction.
        vb = self._cal_vb(ops.stop_gradient(model_output), model_var_values, x, x_t, t)

        loss = mean_flat((noise - model_output) ** 2) + vb
        loss = loss.mean()
        return loss


class UnconditionalModelWithLoss(NetworkWithLoss):
    def construct(self, x: ms.Tensor):
        return super().construct(x, labels=None, text_tokens=None)


class ClassConditionedModelWithLoss(NetworkWithLoss):
    def construct(self, x: ms.Tensor, labels: ms.Tensor):
        return super().construct(x, labels=labels, text_tokens=None)


class TextConditionedModelWithLoss(NetworkWithLoss):
    def construct(self, x: ms.Tensor, text_tokens: ms.Tensor):
        return super().construct(x, labels=None, text_tokens=text_tokens)


def get_model_with_loss(condition):
    if condition is None:
        return UnconditionalModelWithLoss
    elif condition == "class":
        return ClassConditionedModelWithLoss
    elif condition == "text":
        return TextConditionedModelWithLoss
    else:
        raise NotImplementedError
