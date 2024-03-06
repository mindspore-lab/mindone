from diffusion import SpacedDiffusion
from diffusion.diffusion_utils import _extract_into_tensor, discretized_gaussian_log_likelihood, mean_flat, normal_kl

import mindspore as ms
from mindspore import nn, ops


class NetworkWithLoss(nn.Cell):
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

    def construct(self, x: ms.Tensor, text_tokens: ms.Tensor, labels: ms.Tensor, **kwargs):
        """
        Diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames or images, resized and normalized to shape [bs, F, 3, 256, 256] or [bs, 3, 256, 256]
            text: text tokens padded to fixed shape [bs, 77], optional
            labels: class label ids [bs, ], optional

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume unet3d input/output shape: (b c f h w)
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
        # _vb_terms_bpd(model=lambda *_: frozen_out, x_start=x, x_t=x_t, t=t, clip_denoised=False) begin
        true_mean, _, true_log_variance_clipped = self.diffusion.q_posterior_mean_variance(x_start=x, x_t=x_t, t=t)
        # p_mean_variance(model=lambda *_: frozen_out, x_t, t, clip_denoised=False) begin
        min_log = _extract_into_tensor(self.diffusion.posterior_log_variance_clipped, t, x_t.shape)
        max_log = _extract_into_tensor(ops.log(self.diffusion.betas), t, x_t.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        pred_xstart = self.diffusion.predict_xstart_from_eps(x_t=x_t, t=t, eps=ops.stop_gradient(model_output))
        model_mean, _, _ = self.diffusion.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
        # assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        # p_mean_variance end
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = mean_flat(kl) / 0.693147  # np.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(x, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = mean_flat(decoder_nll) / 0.693147  # np.log(2.0)
        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        vb = ops.where((t == 0), decoder_nll.to(kl.dtype), kl)
        # _vb_terms_bpd end

        loss = mean_flat((noise - model_output) ** 2) + vb
        loss = loss.mean()
        return loss


class DiTWithLoss(NetworkWithLoss):
    def apply_model(self, x_t, t, y=None, **kwargs):
        return self.network(x_t, t, y=y)
